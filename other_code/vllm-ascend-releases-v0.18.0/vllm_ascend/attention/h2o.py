#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from typing import Any, Sequence

import torch
from vllm.logger import logger


class H2OBlockPruner:
    """Build compact decode block tables for Ascend paged KV attention.

    H2O selects high-score historical tokens plus recent tokens. Ascend FIA/PA
    kernels consume paged KV through block tables and do not expose attention
    probabilities to Python, so this pruner applies the same heavy+recent budget
    at block granularity. When request ids are available, it keeps a lightweight
    per-request block score so blocks that have been retained repeatedly can stay
    in the heavy set after they leave the recent window.
    """

    def __init__(self):
        self._scores: dict[Any, list[float]] = {}
        self._last_seq_lens: dict[Any, int] = {}
        self._decode_steps: dict[Any, int] = {}
        self._selection_cache: dict[Any, tuple[tuple[Any, ...], tuple[int, ...]]] = {}
        self._compact_metadata_cache: tuple[tuple[Any, ...], torch.Tensor] | None = None
        self._debug_step = 0
        self._debug_seen_pruned = False

    def apply(
        self,
        *,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        config: Any,
        request_ids: Sequence[Any] | None = None,
        seq_lens_list: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if seq_lens_list is None:
            resolved_seq_lens = [] if seq_lens is None else [int(x) for x in seq_lens.tolist()]
        else:
            resolved_seq_lens = [int(x) for x in seq_lens_list]
        original_seq_lens = list(resolved_seq_lens)

        if block_tables is None or seq_lens is None:
            return block_tables, seq_lens, resolved_seq_lens

        if block_size <= 0:
            return block_tables, seq_lens, resolved_seq_lens

        valid_block_counts = [
            math.ceil(seq_len / block_size) if seq_len > 0 else 0 for seq_len in resolved_seq_lens
        ]
        if self._can_keep_original_metadata(
            resolved_seq_lens,
            valid_block_counts,
            config,
            request_ids,
        ):
            self._advance_warmup_decode_steps(resolved_seq_lens, valid_block_counts, config, request_ids)
            return block_tables, seq_lens, resolved_seq_lens

        changed = False
        debug_log = bool(getattr(config, "debug_log", False))
        total_original_blocks = 0
        total_kept_blocks = 0
        planned_kept_blocks = 0
        sample_requests: list[str] = []
        selected_block_rows: list[Sequence[int] | None] = []
        score_updates: list[tuple[int, int, int, Sequence[int] | None, bool]] = []
        prune_candidates: list[tuple[int, int, int, int, int, int]] = []

        for req_index, seq_len in enumerate(resolved_seq_lens):
            if seq_len <= 0:
                selected_block_rows.append([0])
                continue

            valid_blocks = valid_block_counts[req_index]
            total_original_blocks += valid_blocks
            if seq_len < config.min_seq_len:
                selected_block_rows.append(list(range(valid_blocks)))
                total_kept_blocks += valid_blocks
                planned_kept_blocks += valid_blocks
                if debug_log:
                    self._append_debug_sample(
                        sample_requests,
                        config,
                        request_ids,
                        req_index,
                        seq_len,
                        seq_len,
                        valid_blocks,
                        valid_blocks,
                        "short",
                    )
                continue
            if self._should_keep_full_decode_step(req_index, config, request_ids):
                selected_block_rows.append(list(range(valid_blocks)))
                total_kept_blocks += valid_blocks
                planned_kept_blocks += valid_blocks
                score_updates.append((req_index, seq_len, valid_blocks, None, True))
                if debug_log:
                    self._append_debug_sample(
                        sample_requests,
                        config,
                        request_ids,
                        req_index,
                        seq_len,
                        seq_len,
                        valid_blocks,
                        valid_blocks,
                        "decode-warmup",
                    )
                continue
            if self._should_keep_full_context(valid_blocks, config):
                selected_block_rows.append(list(range(valid_blocks)))
                total_kept_blocks += valid_blocks
                planned_kept_blocks += valid_blocks
                if debug_log:
                    self._append_debug_sample(
                        sample_requests,
                        config,
                        request_ids,
                        req_index,
                        seq_len,
                        seq_len,
                        valid_blocks,
                        valid_blocks,
                        "precision-full",
                    )
                continue

            heavy_blocks, recent_blocks = self._resolve_budgets(seq_len, valid_blocks, block_size, config)
            block_cap = self._resolve_block_cap(valid_blocks, config)
            if block_cap is not None:
                recent_blocks = min(recent_blocks, block_cap)
                heavy_blocks = min(heavy_blocks, max(block_cap - recent_blocks, 0))
            heavy_blocks, recent_blocks = self._apply_decode_budget_taper(
                req_index,
                valid_blocks,
                heavy_blocks,
                recent_blocks,
                config,
                request_ids,
            )

            if heavy_blocks + recent_blocks >= valid_blocks:
                selected_block_rows.append(list(range(valid_blocks)))
                total_kept_blocks += valid_blocks
                planned_kept_blocks += valid_blocks
                score_updates.append((req_index, seq_len, valid_blocks, range(valid_blocks), False))
                if debug_log:
                    self._append_debug_sample(
                        sample_requests,
                        config,
                        request_ids,
                        req_index,
                        seq_len,
                        seq_len,
                        valid_blocks,
                        valid_blocks,
                        "budget>=context",
                    )
                continue

            selected_block_rows.append(None)
            planned_kept_blocks += min(valid_blocks, max(heavy_blocks + recent_blocks, 1))
            prune_candidates.append((
                len(selected_block_rows) - 1,
                req_index,
                seq_len,
                valid_blocks,
                heavy_blocks,
                recent_blocks,
            ))
            changed = True

        if not changed:
            if debug_log:
                self._log_debug_summary(
                    config,
                    block_size,
                    len(resolved_seq_lens),
                    total_original_blocks,
                    total_kept_blocks,
                    changed,
                    sample_requests,
                )
            return block_tables, seq_lens, resolved_seq_lens
        if not self._has_enough_pruning_savings(total_original_blocks, planned_kept_blocks, config):
            skip_updates = list(score_updates)
            skip_updates.extend(
                (req_index, seq_len, valid_blocks, None, True)
                for _, req_index, seq_len, valid_blocks, _, _ in prune_candidates
            )
            self._touch_decode_steps_for_skipped_pruning(skip_updates, request_ids)
            if debug_log:
                logger.info(
                    "[H2O] skipped compact metadata before block selection because planned prune ratio %.4f "
                    "is below min_prune_ratio %.4f",
                    self._prune_ratio(total_original_blocks, planned_kept_blocks),
                    getattr(config, "min_prune_ratio", 0.0),
                )
                self._log_debug_summary(
                    config,
                    block_size,
                    len(resolved_seq_lens),
                    total_original_blocks,
                    planned_kept_blocks,
                    False,
                    sample_requests,
                )
            return block_tables, seq_lens, original_seq_lens

        for row_pos, req_index, seq_len, valid_blocks, heavy_blocks, recent_blocks in prune_candidates:
            selected, cache_hit = self._select_blocks(
                req_index,
                seq_len,
                valid_blocks,
                heavy_blocks,
                recent_blocks,
                config,
                request_ids,
            )
            compact_len = self._selected_token_count(selected, seq_len, valid_blocks, block_size)
            selected_block_rows[row_pos] = selected
            resolved_seq_lens[req_index] = compact_len
            total_kept_blocks += len(selected)
            score_updates.append((req_index, seq_len, valid_blocks, selected, cache_hit))
            if debug_log:
                self._append_debug_sample(
                    sample_requests,
                    config,
                    request_ids,
                    req_index,
                    seq_len,
                    compact_len,
                    valid_blocks,
                    len(selected),
                    "pruned",
                )

        if debug_log:
            self._log_debug_summary(
                config,
                block_size,
                len(resolved_seq_lens),
                total_original_blocks,
                total_kept_blocks,
                changed,
                sample_requests,
            )

        self._apply_score_updates(score_updates, config, request_ids)
        compact_rows = [row if row is not None else (0,) for row in selected_block_rows]
        return self._build_compact_metadata(block_tables, seq_lens, compact_rows, resolved_seq_lens, config)

    def _can_keep_original_metadata(
        self,
        seq_lens: Sequence[int],
        valid_block_counts: Sequence[int],
        config: Any,
        request_ids: Sequence[Any] | None,
    ) -> bool:
        if not seq_lens:
            return True
        for req_index, (seq_len, valid_blocks) in enumerate(zip(seq_lens, valid_block_counts)):
            if seq_len <= 0:
                continue
            if seq_len < config.min_seq_len:
                continue
            if H2OBlockPruner._should_keep_full_context(valid_blocks, config):
                continue
            if self._should_keep_full_decode_step(req_index, config, request_ids):
                continue
            return False
        return True

    def _advance_warmup_decode_steps(
        self,
        seq_lens: Sequence[int],
        valid_block_counts: Sequence[int],
        config: Any,
        request_ids: Sequence[Any] | None,
    ) -> None:
        for req_index, (seq_len, valid_blocks) in enumerate(zip(seq_lens, valid_block_counts)):
            if seq_len <= 0 or seq_len < config.min_seq_len:
                continue
            if self._should_keep_full_context(valid_blocks, config):
                continue
            if self._should_keep_full_decode_step(req_index, config, request_ids):
                self._touch_decode_step(req_index, seq_len, valid_blocks, request_ids)

    def _apply_score_updates(
        self,
        score_updates: Sequence[tuple[int, int, int, Sequence[int] | None, bool]],
        config: Any,
        request_ids: Sequence[Any] | None,
    ) -> None:
        for req_index, seq_len, valid_blocks, selected, cache_hit in score_updates:
            if selected is None or (cache_hit and not getattr(config, "score_update_on_cache_hit", False)):
                self._touch_decode_step(req_index, seq_len, valid_blocks, request_ids)
            else:
                self._update_scores(req_index, seq_len, valid_blocks, selected, config, request_ids)

    def _touch_decode_steps_for_skipped_pruning(
        self,
        score_updates: Sequence[tuple[int, int, int, Sequence[int] | None, bool]],
        request_ids: Sequence[Any] | None,
    ) -> None:
        for req_index, seq_len, valid_blocks, _, _ in score_updates:
            self._touch_decode_step(req_index, seq_len, valid_blocks, request_ids)

    def _build_compact_metadata(
        self,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        selected_block_rows: Sequence[Sequence[int]],
        compact_seq_lens: Sequence[int],
        config: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        max_selected_blocks = max((len(row) for row in selected_block_rows), default=1)
        max_selected_blocks = max(max_selected_blocks, 1)
        metadata_width = self._resolve_compact_metadata_width(
            max_selected_blocks,
            block_tables.shape[1],
            config,
        )
        selected_rows = tuple(tuple(row) if row else (0,) for row in selected_block_rows)
        cache_key = (str(block_tables.device), block_tables.shape[0], block_tables.shape[1], metadata_width,
                     selected_rows)
        cached = self._compact_metadata_cache
        if cached is not None and cached[0] == cache_key:
            gather_indices = cached[1]
        else:
            gather_rows = []
            for selected in selected_rows:
                row = list(selected)
                if len(row) < metadata_width:
                    row = row + [0] * (metadata_width - len(row))
                gather_rows.append(row)

            gather_indices = torch.as_tensor(
                gather_rows,
                dtype=torch.long,
                device=block_tables.device,
            )
            self._compact_metadata_cache = (cache_key, gather_indices)
        new_block_tables = block_tables.gather(1, gather_indices)
        compact_seq_lens_list = [int(seq_len) for seq_len in compact_seq_lens]
        new_seq_lens = torch.tensor(
            compact_seq_lens_list,
            dtype=seq_lens.dtype,
            device=seq_lens.device,
        )
        return new_block_tables, new_seq_lens, compact_seq_lens_list

    @staticmethod
    def _resolve_compact_metadata_width(max_selected_blocks: int, block_table_width: int, config: Any) -> int:
        metadata_width = max_selected_blocks
        if (
            getattr(config, "adaptive_budget", True)
            and getattr(config, "max_blocks", None) is not None
            and getattr(config, "adaptive_precision_ratio", 0.0) > 0
        ):
            precision_max_blocks = getattr(config, "adaptive_precision_max_blocks", None)
            if precision_max_blocks is not None:
                metadata_width = max(metadata_width, int(precision_max_blocks))
        return max(1, min(metadata_width, block_table_width))

    @staticmethod
    def _has_enough_pruning_savings(total_original_blocks: int, total_kept_blocks: int, config: Any) -> bool:
        min_prune_ratio = getattr(config, "min_prune_ratio", 0.0)
        if min_prune_ratio <= 0:
            return True
        return H2OBlockPruner._prune_ratio(total_original_blocks, total_kept_blocks) >= min_prune_ratio

    @staticmethod
    def _prune_ratio(total_original_blocks: int, total_kept_blocks: int) -> float:
        if total_original_blocks <= 0:
            return 0.0
        return max(total_original_blocks - total_kept_blocks, 0) / total_original_blocks

    def _log_debug_summary(
        self,
        config: Any,
        block_size: int,
        batch_size: int,
        total_original_blocks: int,
        total_kept_blocks: int,
        changed: bool,
        sample_requests: Sequence[str],
    ) -> None:
        self._debug_step += 1
        should_log = self._debug_step == 1 or self._debug_step % config.debug_interval == 0
        if changed and not self._debug_seen_pruned:
            should_log = True
            self._debug_seen_pruned = True
        if not should_log:
            return

        pruned_blocks = max(total_original_blocks - total_kept_blocks, 0)
        keep_ratio = total_kept_blocks / total_original_blocks if total_original_blocks else 1.0
        rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "unknown"))
        status = "active" if changed else "enabled_no_prune"
        logger.info(
            "[H2O][rank=%s] decode pruning %s: step=%d batch=%d block_size=%d "
            "original_blocks=%d kept_blocks=%d pruned_blocks=%d keep_ratio=%.4f samples=%s",
            rank,
            status,
            self._debug_step,
            batch_size,
            block_size,
            total_original_blocks,
            total_kept_blocks,
            pruned_blocks,
            keep_ratio,
            "; ".join(sample_requests) if sample_requests else "[]",
        )

    def _append_debug_sample(
        self,
        sample_requests: list[str],
        config: Any,
        request_ids: Sequence[Any] | None,
        req_index: int,
        original_len: int,
        compact_len: int,
        original_blocks: int,
        kept_blocks: int,
        reason: str,
    ) -> None:
        debug_sample_requests = getattr(config, "debug_sample_requests", 3)
        if len(sample_requests) >= debug_sample_requests:
            return
        request_id = self._get_request_id(req_index, request_ids)
        if request_id is None:
            request_id = req_index
        sample_requests.append(
            f"req={request_id} len={original_len}->{compact_len} "
            f"blocks={original_blocks}->{kept_blocks} reason={reason}"
        )

    @staticmethod
    def _resolve_budgets(
        seq_len: int,
        valid_blocks: int,
        block_size: int,
        config: Any,
    ) -> tuple[int, int]:
        heavy_blocks = H2OBlockPruner._blocks_from_budget(
            seq_len,
            valid_blocks,
            block_size,
            config.heavy_ratio,
            config.heavy_blocks,
        )
        recent_blocks = H2OBlockPruner._blocks_from_budget(
            seq_len,
            valid_blocks,
            block_size,
            config.recent_ratio,
            config.recent_blocks,
        )
        if H2OBlockPruner._should_expand_fixed_budget(config):
            min_keep_blocks = heavy_blocks + recent_blocks
            if config.adaptive_min_keep_ratio > 0:
                min_keep_blocks = max(
                    min_keep_blocks,
                    math.ceil(valid_blocks * config.adaptive_min_keep_ratio),
                )
            precision_blocks = H2OBlockPruner._adaptive_precision_blocks(valid_blocks, config)
            if precision_blocks is not None:
                min_keep_blocks = max(min_keep_blocks, precision_blocks)
            elif config.max_blocks is not None and config.adaptive_min_keep_ratio > 0:
                min_keep_blocks = min(min_keep_blocks, config.max_blocks)
            if min_keep_blocks > heavy_blocks + recent_blocks:
                heavy_blocks += min_keep_blocks - heavy_blocks - recent_blocks
        return heavy_blocks, recent_blocks

    @staticmethod
    def _resolve_block_cap(valid_blocks: int, config: Any) -> int | None:
        max_blocks = getattr(config, "max_blocks", None)
        if max_blocks is None:
            return None
        block_cap = min(max_blocks, valid_blocks)
        precision_blocks = H2OBlockPruner._adaptive_precision_blocks(valid_blocks, config)
        if precision_blocks is not None:
            block_cap = max(block_cap, precision_blocks)
        return min(block_cap, valid_blocks)

    @staticmethod
    def _adaptive_precision_blocks(valid_blocks: int, config: Any) -> int | None:
        if not getattr(config, "adaptive_budget", True):
            return None
        if getattr(config, "max_blocks", None) is None:
            return None

        precision_ratio = getattr(config, "adaptive_precision_ratio", 0.0)
        if precision_ratio <= 0:
            return None
        if H2OBlockPruner._should_keep_full_context(valid_blocks, config):
            return valid_blocks

        precision_blocks = math.ceil(valid_blocks * precision_ratio)
        precision_max_blocks = getattr(config, "adaptive_precision_max_blocks", None)
        if precision_max_blocks is not None:
            precision_blocks = min(precision_blocks, precision_max_blocks)
        precision_blocks = max(precision_blocks, config.max_blocks)
        return min(precision_blocks, valid_blocks)

    @staticmethod
    def _should_expand_fixed_budget(config: Any) -> bool:
        if not getattr(config, "adaptive_budget", True):
            return False
        return config.heavy_blocks is not None or config.recent_blocks is not None

    @staticmethod
    def _should_keep_full_context(valid_blocks: int, config: Any) -> bool:
        if valid_blocks <= 0:
            return True
        if not getattr(config, "adaptive_budget", True):
            return False
        if getattr(config, "max_blocks", None) is None:
            return False
        if getattr(config, "adaptive_precision_ratio", 0.0) <= 0:
            return False

        precision_max_blocks = getattr(config, "adaptive_precision_max_blocks", None)
        return precision_max_blocks is not None and valid_blocks <= precision_max_blocks

    @staticmethod
    def _blocks_from_budget(
        seq_len: int,
        valid_blocks: int,
        block_size: int,
        ratio: float,
        explicit_blocks: int | None,
    ) -> int:
        if explicit_blocks is not None:
            return min(explicit_blocks, valid_blocks)
        if ratio <= 0:
            return 0
        token_budget = max(1, int(seq_len * ratio))
        return min(math.ceil(token_budget / block_size), valid_blocks)

    def _apply_decode_budget_taper(
        self,
        req_index: int,
        valid_blocks: int,
        heavy_blocks: int,
        recent_blocks: int,
        config: Any,
        request_ids: Sequence[Any] | None,
    ) -> tuple[int, int]:
        fast_blocks = getattr(config, "decode_budget_fast_blocks", None)
        fast_ratio = getattr(config, "decode_budget_fast_ratio", 0.0)
        taper_steps = getattr(config, "decode_budget_taper_steps", 0)
        if (fast_blocks is None and fast_ratio <= 0) or taper_steps <= 0:
            return heavy_blocks, recent_blocks
        if self._should_keep_full_context(valid_blocks, config):
            return heavy_blocks, recent_blocks

        current_total = heavy_blocks + recent_blocks
        if current_total <= 0:
            return heavy_blocks, recent_blocks

        start_step = getattr(config, "decode_budget_taper_start_step", 0)
        decode_step = max(self._get_decode_step(req_index, request_ids) - start_step, 0)
        if decode_step <= 0:
            return heavy_blocks, recent_blocks

        if fast_blocks is None:
            fast_target = math.ceil(valid_blocks * fast_ratio)
        else:
            fast_target = fast_blocks
        max_blocks = getattr(config, "max_blocks", None)
        if max_blocks is not None:
            fast_target = min(fast_target, max_blocks)

        min_heavy_blocks = min(heavy_blocks, getattr(config, "sink_blocks", 1))
        min_total = min(valid_blocks, recent_blocks + min_heavy_blocks)
        fast_target = min(max(fast_target, min_total), current_total, valid_blocks)
        if fast_target >= current_total:
            return heavy_blocks, recent_blocks

        progress = min(decode_step / taper_steps, 1.0)
        tapered_total = math.ceil(current_total - (current_total - fast_target) * progress)
        tapered_total = min(max(tapered_total, min_total), current_total)
        if tapered_total >= current_total:
            return heavy_blocks, recent_blocks

        tapered_heavy_blocks = max(tapered_total - recent_blocks, min_heavy_blocks)
        return tapered_heavy_blocks, recent_blocks

    def _should_keep_full_decode_step(
        self,
        req_index: int,
        config: Any,
        request_ids: Sequence[Any] | None,
    ) -> bool:
        warmup_steps = getattr(config, "decode_full_attention_steps", 0)
        if warmup_steps <= 0:
            return False
        if self._get_request_id(req_index, request_ids) is None:
            return False
        return self._get_decode_step(req_index, request_ids) < warmup_steps

    def _select_blocks(
        self,
        req_index: int,
        seq_len: int,
        valid_blocks: int,
        heavy_blocks: int,
        recent_blocks: int,
        config: Any,
        request_ids: Sequence[Any] | None,
    ) -> tuple[Sequence[int], bool]:
        cache_key = self._selection_cache_key(
            req_index,
            valid_blocks,
            heavy_blocks,
            recent_blocks,
            config,
            request_ids,
        )
        cached = self._get_cached_selection(req_index, cache_key, config, request_ids)
        if cached is not None:
            return cached, True

        recent_start = max(valid_blocks - recent_blocks, 0)
        recent = list(range(recent_start, valid_blocks))
        heavy_candidate_end = recent_start
        if heavy_blocks <= 0 or heavy_candidate_end <= 0:
            return self._cache_selection(req_index, cache_key, recent, valid_blocks, request_ids), False

        sink_blocks = min(getattr(config, "sink_blocks", 1), heavy_blocks, heavy_candidate_end)
        sink = list(range(sink_blocks))
        remaining_heavy_blocks = heavy_blocks - len(sink)
        heavy_candidate_start = sink_blocks
        if remaining_heavy_blocks <= 0 or heavy_candidate_start >= heavy_candidate_end:
            return self._cache_selection(req_index, cache_key, sink + recent, valid_blocks, request_ids), False

        history_cluster_size = getattr(config, "history_cluster_size", 1)
        scores = self._get_scores(req_index, seq_len, valid_blocks, request_ids)
        has_score_signal = scores is not None and any(
            score > 0 for score in scores[heavy_candidate_start:heavy_candidate_end])
        if not has_score_signal:
            heavy = sink + self._clustered_evenly_spaced_blocks(
                heavy_candidate_start,
                heavy_candidate_end,
                remaining_heavy_blocks,
                history_cluster_size,
                set(sink),
            )
        else:
            anchor_budget = min(
                remaining_heavy_blocks,
                math.ceil(remaining_heavy_blocks * getattr(config, "anchor_ratio", 0.25)),
            )
            anchor_blocks = self._anchor_count_for_cluster_budget(anchor_budget, history_cluster_size)
            anchors = self._score_guided_anchor_blocks(
                heavy_candidate_start,
                heavy_candidate_end,
                anchor_blocks,
                scores,
            )
            clustered_anchors = self._cluster_blocks_around_anchors(
                heavy_candidate_start,
                heavy_candidate_end,
                anchors,
                anchor_budget,
                history_cluster_size,
                set(sink),
            )
            reserved = set(sink) | set(clustered_anchors)
            explore_ratio = getattr(config, "score_explore_ratio", 0.0)
            explore_blocks = 0
            if explore_ratio > 0:
                explore_blocks = min(
                    max(remaining_heavy_blocks - len(clustered_anchors), 0),
                    math.ceil(remaining_heavy_blocks * explore_ratio),
                )
            exploration = self._rotating_evenly_spaced_blocks(
                heavy_candidate_start,
                heavy_candidate_end,
                explore_blocks,
                self._get_decode_step(req_index, request_ids),
                reserved,
            )
            reserved.update(exploration)
            coverage_ratio = getattr(config, "score_coverage_ratio", 0.0)
            coverage_blocks = 0
            if coverage_ratio > 0:
                coverage_blocks = min(
                    max(heavy_blocks - len(reserved), 0),
                    math.ceil(remaining_heavy_blocks * coverage_ratio),
                )
            coverage = self._coverage_blocks(
                heavy_candidate_start,
                heavy_candidate_end,
                coverage_blocks,
                reserved,
            )
            reserved.update(coverage)
            score_blocks = max(heavy_blocks - len(reserved), 0)
            if score_blocks > 0:
                ranked_candidates = [
                    index for index in range(heavy_candidate_start, heavy_candidate_end) if index not in reserved
                ]
                ranked = sorted(ranked_candidates, key=lambda index: (-scores[index], index))
                reserved.update(ranked[:score_blocks])
            heavy = sorted(reserved)
        return self._cache_selection(req_index, cache_key, heavy + recent, valid_blocks, request_ids), False

    def _selection_cache_key(
        self,
        req_index: int,
        valid_blocks: int,
        heavy_blocks: int,
        recent_blocks: int,
        config: Any,
        request_ids: Sequence[Any] | None,
    ) -> tuple[Any, ...] | None:
        request_id = self._get_request_id(req_index, request_ids)
        if request_id is None:
            return None
        return (
            valid_blocks,
            heavy_blocks,
            recent_blocks,
            getattr(config, "sink_blocks", 1),
            getattr(config, "anchor_ratio", 0.25),
            getattr(config, "score_explore_ratio", 0.0),
            getattr(config, "score_coverage_ratio", 0.0),
            getattr(config, "history_cluster_size", 1),
        )

    def _get_cached_selection(
        self,
        req_index: int,
        cache_key: tuple[Any, ...] | None,
        config: Any,
        request_ids: Sequence[Any] | None,
    ) -> tuple[int, ...] | None:
        if cache_key is None:
            return None
        refresh_interval = getattr(config, "selection_refresh_interval", 1)
        if refresh_interval <= 1:
            return None
        decode_step = self._get_decode_step(req_index, request_ids)
        if decode_step % refresh_interval == 0:
            return None
        request_id = self._get_request_id(req_index, request_ids)
        cached = self._selection_cache.get(request_id)
        if cached is None:
            return None
        cached_key, cached_selection = cached
        if cached_key != cache_key:
            return None
        return cached_selection

    def _cache_selection(
        self,
        req_index: int,
        cache_key: tuple[Any, ...] | None,
        selected: Sequence[int],
        valid_blocks: int,
        request_ids: Sequence[Any] | None,
    ) -> tuple[int, ...]:
        selected_tuple = self._ensure_current_block_selected(selected, valid_blocks)
        if cache_key is None:
            return selected_tuple
        request_id = self._get_request_id(req_index, request_ids)
        if request_id is not None:
            self._selection_cache[request_id] = (cache_key, selected_tuple)
        return selected_tuple

    @staticmethod
    def _ensure_current_block_selected(selected: Sequence[int], valid_blocks: int) -> tuple[int, ...]:
        current_block = valid_blocks - 1
        if current_block < 0:
            return (0,)
        if not selected:
            return (current_block,)
        selected_tuple = tuple(selected)
        if selected_tuple[-1] == current_block:
            return selected_tuple
        if current_block not in selected_tuple:
            selected_tuple = selected_tuple + (current_block,)
        return tuple(sorted(set(selected_tuple)))

    @staticmethod
    def _score_guided_anchor_blocks(start: int, end: int, count: int, scores: Sequence[float]) -> list[int]:
        if count <= 0 or start >= end:
            return []
        span = end - start
        if count >= span:
            return list(range(start, end))

        anchors: list[int] = []
        seen: set[int] = set()
        for bucket in range(count):
            bucket_start = start + bucket * span // count
            bucket_end = start + (bucket + 1) * span // count
            if bucket_end <= bucket_start:
                bucket_end = bucket_start + 1
            bucket_end = min(bucket_end, end)
            center = bucket_start + (bucket_end - bucket_start) // 2
            best = max(
                range(bucket_start, bucket_end),
                key=lambda index: (scores[index], -abs(index - center), -index),
            )
            anchors.append(best)
            seen.add(best)

        if len(anchors) < count:
            for block in H2OBlockPruner._evenly_spaced_blocks(start, end, count):
                if block in seen:
                    continue
                anchors.append(block)
                seen.add(block)
                if len(anchors) == count:
                    break
        return sorted(anchors)

    @staticmethod
    def _anchor_count_for_cluster_budget(block_budget: int, cluster_size: int) -> int:
        if block_budget <= 0:
            return 0
        cluster_size = max(int(cluster_size), 1)
        return max(1, math.ceil(block_budget / cluster_size))

    @staticmethod
    def _clustered_evenly_spaced_blocks(
        start: int,
        end: int,
        count: int,
        cluster_size: int,
        excluded: set[int],
    ) -> list[int]:
        if cluster_size <= 1:
            return H2OBlockPruner._coverage_blocks(start, end, count, excluded)
        anchor_count = H2OBlockPruner._anchor_count_for_cluster_budget(count, cluster_size)
        anchors = H2OBlockPruner._evenly_spaced_blocks(start, end, anchor_count)
        return H2OBlockPruner._cluster_blocks_around_anchors(
            start,
            end,
            anchors,
            count,
            cluster_size,
            excluded,
        )

    @staticmethod
    def _cluster_blocks_around_anchors(
        start: int,
        end: int,
        anchors: Sequence[int],
        budget: int,
        cluster_size: int,
        excluded: set[int],
    ) -> list[int]:
        if budget <= 0 or start >= end:
            return []
        cluster_size = max(int(cluster_size), 1)
        selected: list[int] = []
        seen = set(excluded)

        offsets = [0]
        distance = 1
        while len(offsets) < cluster_size:
            offsets.append(distance)
            if len(offsets) < cluster_size:
                offsets.append(-distance)
            distance += 1

        for anchor in anchors:
            for offset in offsets:
                candidate = anchor + offset
                if candidate < start or candidate >= end or candidate in seen:
                    continue
                selected.append(candidate)
                seen.add(candidate)
                if len(selected) == budget:
                    return sorted(selected)

        if len(selected) < budget:
            for candidate in H2OBlockPruner._evenly_spaced_blocks(start, end, min(end - start, budget * 2)):
                if candidate in seen:
                    continue
                selected.append(candidate)
                seen.add(candidate)
                if len(selected) == budget:
                    return sorted(selected)
        if len(selected) < budget:
            for candidate in range(start, end):
                if candidate in seen:
                    continue
                selected.append(candidate)
                seen.add(candidate)
                if len(selected) == budget:
                    break
        return sorted(selected)

    @staticmethod
    def _evenly_spaced_blocks(start: int, end: int, count: int) -> list[int]:
        if count <= 0 or start >= end:
            return []
        span = end - start
        if count >= span:
            return list(range(start, end))
        if count == 1:
            return [start + span // 2]

        blocks: list[int] = []
        seen: set[int] = set()
        for index in range(count):
            block = start + min(span - 1, int((index + 0.5) * span / count))
            if block not in seen:
                blocks.append(block)
                seen.add(block)
        if len(blocks) < count:
            for block in range(start, end):
                if block in seen:
                    continue
                blocks.append(block)
                seen.add(block)
                if len(blocks) == count:
                    break
        return sorted(blocks)

    @staticmethod
    def _coverage_blocks(start: int, end: int, count: int, excluded: set[int]) -> list[int]:
        if count <= 0 or start >= end:
            return []

        blocks: list[int] = []
        seen = set(excluded)

        def append_candidates(candidates: Sequence[int]) -> None:
            for block in candidates:
                if block in seen:
                    continue
                blocks.append(block)
                seen.add(block)
                if len(blocks) == count:
                    break

        append_candidates(H2OBlockPruner._evenly_spaced_blocks(start, end, count))
        if len(blocks) < count:
            append_candidates(H2OBlockPruner._evenly_spaced_blocks(start, end, min(end - start, count * 2)))
        if len(blocks) < count:
            append_candidates(range(start, end))
        return sorted(blocks)

    @staticmethod
    def _rotating_evenly_spaced_blocks(
        start: int,
        end: int,
        count: int,
        phase: int,
        excluded: set[int],
    ) -> list[int]:
        if count <= 0 or start >= end:
            return []
        span = end - start
        if count >= span:
            return [block for block in range(start, end) if block not in excluded][:count]

        blocks: list[int] = []
        seen = set(excluded)
        for bucket in range(count):
            bucket_start = start + bucket * span // count
            bucket_end = start + (bucket + 1) * span // count
            if bucket_end <= bucket_start:
                bucket_end = bucket_start + 1
            bucket_end = min(bucket_end, end)
            bucket_span = bucket_end - bucket_start
            for attempt in range(bucket_span):
                block = bucket_start + (phase + attempt) % bucket_span
                if block in seen:
                    continue
                blocks.append(block)
                seen.add(block)
                break

        if len(blocks) < count:
            for block in H2OBlockPruner._evenly_spaced_blocks(start, end, count):
                if block in seen:
                    continue
                blocks.append(block)
                seen.add(block)
                if len(blocks) == count:
                    break
        if len(blocks) < count:
            for block in range(start, end):
                if block in seen:
                    continue
                blocks.append(block)
                if len(blocks) == count:
                    break
        return sorted(blocks)

    def _get_scores(
        self,
        req_index: int,
        seq_len: int,
        valid_blocks: int,
        request_ids: Sequence[Any] | None,
    ) -> list[float] | None:
        request_id = self._get_request_id(req_index, request_ids)
        if request_id is None:
            return None

        last_seq_len = self._last_seq_lens.get(request_id)
        if last_seq_len is not None and seq_len < last_seq_len:
            self._scores.pop(request_id, None)
            self._decode_steps.pop(request_id, None)
            self._selection_cache.pop(request_id, None)

        scores = self._scores.setdefault(request_id, [])
        if len(scores) < valid_blocks:
            scores.extend([0.0] * (valid_blocks - len(scores)))
        elif len(scores) > valid_blocks:
            del scores[valid_blocks:]
        self._last_seq_lens[request_id] = seq_len
        return scores

    def _update_scores(
        self,
        req_index: int,
        seq_len: int,
        valid_blocks: int,
        selected: Sequence[int],
        config: Any,
        request_ids: Sequence[Any] | None,
    ) -> None:
        scores = self._get_scores(req_index, seq_len, valid_blocks, request_ids)
        if scores is None:
            return
        if config.score_decay < 1.0:
            for index, score in enumerate(scores):
                scores[index] = score * config.score_decay
        recent_start = valid_blocks
        for index in reversed(selected):
            if index == recent_start - 1:
                recent_start -= 1
            elif index < recent_start - 1:
                break
        recent_blocks = valid_blocks - recent_start
        sink_blocks = min(getattr(config, "sink_blocks", 1), valid_blocks)
        for index in selected:
            increment = 1.0
            if index < sink_blocks:
                increment += 0.25
            if recent_blocks > 0 and index >= recent_start:
                recency_rank = index - recent_start + 1
                increment += 0.75 * recency_rank / recent_blocks
            scores[index] += increment
        self._advance_decode_step(req_index, request_ids)

    def _touch_decode_step(
        self,
        req_index: int,
        seq_len: int,
        valid_blocks: int,
        request_ids: Sequence[Any] | None,
    ) -> None:
        request_id = self._get_request_id(req_index, request_ids)
        if request_id is None:
            return

        last_seq_len = self._last_seq_lens.get(request_id)
        if last_seq_len is not None and seq_len < last_seq_len:
            self._scores.pop(request_id, None)
            self._decode_steps.pop(request_id, None)
            self._selection_cache.pop(request_id, None)

        scores = self._scores.get(request_id)
        if scores is not None:
            if len(scores) < valid_blocks:
                scores.extend([0.0] * (valid_blocks - len(scores)))
            elif len(scores) > valid_blocks:
                del scores[valid_blocks:]
        self._last_seq_lens[request_id] = seq_len
        self._advance_decode_step(req_index, request_ids)

    def _advance_decode_step(self, req_index: int, request_ids: Sequence[Any] | None) -> None:
        request_id = self._get_request_id(req_index, request_ids)
        if request_id is not None:
            self._decode_steps[request_id] = self._decode_steps.get(request_id, 0) + 1

    def _get_decode_step(self, req_index: int, request_ids: Sequence[Any] | None) -> int:
        request_id = self._get_request_id(req_index, request_ids)
        if request_id is None:
            return 0
        return self._decode_steps.get(request_id, 0)

    @staticmethod
    def _get_request_id(req_index: int, request_ids: Sequence[Any] | None) -> Any | None:
        if request_ids is None or req_index >= len(request_ids):
            return None
        return request_ids[req_index]

    @staticmethod
    def _selected_token_count(
        selected: Sequence[int],
        seq_len: int,
        valid_blocks: int,
        block_size: int,
    ) -> int:
        last_block_tokens = seq_len - (valid_blocks - 1) * block_size
        if selected and selected[-1] == valid_blocks - 1:
            return (len(selected) - 1) * block_size + last_block_tokens
        total = 0
        for block_index in selected:
            total += last_block_tokens if block_index == valid_blocks - 1 else block_size
        return total
