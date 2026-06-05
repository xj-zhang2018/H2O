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
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if block_tables is None or seq_lens is None:
            seq_lens_list = [] if seq_lens is None else [int(x) for x in seq_lens.tolist()]
            return block_tables, seq_lens, seq_lens_list

        seq_lens_list = [int(x) for x in seq_lens.tolist()]
        if block_size <= 0:
            return block_tables, seq_lens, seq_lens_list

        new_block_tables = torch.zeros_like(block_tables)
        new_seq_lens = torch.empty_like(seq_lens)
        changed = False
        debug_log = bool(getattr(config, "debug_log", False))
        total_original_blocks = 0
        total_kept_blocks = 0
        sample_requests: list[str] = []

        for req_index, seq_len in enumerate(seq_lens_list):
            if seq_len <= 0:
                new_seq_lens[req_index] = 0
                continue

            valid_blocks = math.ceil(seq_len / block_size)
            total_original_blocks += valid_blocks
            if seq_len < config.min_seq_len:
                self._copy_original_row(
                    new_block_tables,
                    block_tables,
                    new_seq_lens,
                    req_index,
                    seq_len,
                )
                total_kept_blocks += valid_blocks
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

            heavy_blocks, recent_blocks = self._resolve_budgets(seq_len, valid_blocks, block_size, config)
            block_cap = self._resolve_block_cap(valid_blocks, config)
            if block_cap is not None:
                recent_blocks = min(recent_blocks, block_cap)
                heavy_blocks = min(heavy_blocks, max(block_cap - recent_blocks, 0))

            if heavy_blocks + recent_blocks >= valid_blocks:
                self._copy_original_row(
                    new_block_tables,
                    block_tables,
                    new_seq_lens,
                    req_index,
                    seq_len,
                )
                total_kept_blocks += valid_blocks
                self._update_scores(req_index, seq_len, valid_blocks, range(valid_blocks), config, request_ids)
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

            selected = self._select_blocks(
                req_index,
                seq_len,
                valid_blocks,
                heavy_blocks,
                recent_blocks,
                config,
                request_ids,
            )
            if not selected:
                selected = [valid_blocks - 1]
            elif selected[-1] != valid_blocks - 1:
                selected.append(valid_blocks - 1)
            selected = sorted(set(selected))

            selected_tensor = torch.tensor(selected, dtype=torch.long, device=block_tables.device)
            selected_blocks = block_tables[req_index].index_select(0, selected_tensor)
            new_block_tables[req_index, : selected_blocks.shape[0]] = selected_blocks

            compact_len = self._selected_token_count(selected, seq_len, valid_blocks, block_size)
            new_seq_lens[req_index] = compact_len
            seq_lens_list[req_index] = compact_len
            total_kept_blocks += len(selected)
            changed = True
            self._update_scores(req_index, seq_len, valid_blocks, selected, config, request_ids)
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
                len(seq_lens_list),
                total_original_blocks,
                total_kept_blocks,
                changed,
                sample_requests,
            )
        if not changed:
            return block_tables, seq_lens, [int(x) for x in seq_lens.tolist()]
        return new_block_tables, new_seq_lens, seq_lens_list

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
    def _copy_original_row(
        new_block_tables: torch.Tensor,
        block_tables: torch.Tensor,
        new_seq_lens: torch.Tensor,
        req_index: int,
        seq_len: int,
    ) -> None:
        new_block_tables[req_index] = block_tables[req_index]
        new_seq_lens[req_index] = seq_len

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

    def _select_blocks(
        self,
        req_index: int,
        seq_len: int,
        valid_blocks: int,
        heavy_blocks: int,
        recent_blocks: int,
        config: Any,
        request_ids: Sequence[Any] | None,
    ) -> list[int]:
        recent_start = max(valid_blocks - recent_blocks, 0)
        recent = list(range(recent_start, valid_blocks))
        heavy_candidate_end = recent_start
        if heavy_blocks <= 0 or heavy_candidate_end <= 0:
            return recent

        sink_blocks = min(getattr(config, "sink_blocks", 1), heavy_blocks, heavy_candidate_end)
        sink = list(range(sink_blocks))
        remaining_heavy_blocks = heavy_blocks - len(sink)
        heavy_candidate_start = sink_blocks
        if remaining_heavy_blocks <= 0 or heavy_candidate_start >= heavy_candidate_end:
            return sink + recent

        scores = self._get_scores(req_index, seq_len, valid_blocks, request_ids)
        has_score_signal = scores is not None and any(
            score > 0 for score in scores[heavy_candidate_start:heavy_candidate_end])
        if not has_score_signal:
            heavy = sink + self._evenly_spaced_blocks(
                heavy_candidate_start,
                heavy_candidate_end,
                remaining_heavy_blocks,
            )
        else:
            anchor_blocks = min(
                remaining_heavy_blocks,
                math.ceil(remaining_heavy_blocks * getattr(config, "anchor_ratio", 0.25)),
            )
            anchors = self._score_guided_anchor_blocks(
                heavy_candidate_start,
                heavy_candidate_end,
                anchor_blocks,
                scores,
            )
            reserved = set(sink) | set(anchors)
            explore_ratio = getattr(config, "score_explore_ratio", 0.0)
            explore_blocks = 0
            if explore_ratio > 0:
                explore_blocks = min(
                    max(remaining_heavy_blocks - len(anchors), 0),
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
            ranked_candidates = [
                index for index in range(heavy_candidate_start, heavy_candidate_end) if index not in reserved
            ]
            ranked = sorted(ranked_candidates, key=lambda index: (-scores[index], index))
            score_blocks = max(heavy_blocks - len(reserved), 0)
            heavy = sorted(reserved | set(ranked[:score_blocks]))
        return heavy + recent

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
        for index in selected:
            scores[index] += 1.0
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
        total = 0
        for block_index in selected:
            total += last_block_tokens if block_index == valid_blocks - 1 else block_size
        return total
