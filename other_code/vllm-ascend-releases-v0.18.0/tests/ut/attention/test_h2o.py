from dataclasses import dataclass

import torch

from vllm_ascend.attention.h2o import H2OBlockPruner


@dataclass
class H2OConfigStub:
    enabled: bool = True
    heavy_ratio: float = 0.1
    recent_ratio: float = 0.1
    heavy_blocks: int | None = None
    recent_blocks: int | None = None
    max_blocks: int | None = None
    min_seq_len: int = 0
    score_decay: float = 1.0
    adaptive_budget: bool = True
    adaptive_min_keep_ratio: float = 0.1
    adaptive_precision_ratio: float = 0.6
    adaptive_precision_max_blocks: int | None = 96
    sink_blocks: int = 1
    anchor_ratio: float = 0.25
    score_explore_ratio: float = 0.2
    score_coverage_ratio: float = 0.35
    min_prune_ratio: float = 0.0
    history_cluster_size: int = 1
    decode_full_attention_steps: int = 0
    decode_budget_fast_blocks: int | None = None
    decode_budget_fast_ratio: float = 0.45
    decode_budget_taper_steps: int = 256
    decode_budget_taper_start_step: int = 64
    selection_refresh_interval: int = 4
    score_update_on_cache_hit: bool = False
    debug_log: bool = False
    debug_interval: int = 1
    debug_sample_requests: int = 3


def test_h2o_pruner_keeps_heavy_and_recent_blocks():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(heavy_blocks=1, recent_blocks=2)
    block_tables = torch.tensor([[10, 11, 12, 13, 14, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([5 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables[0, :3].tolist() == [10, 13, 14]
    assert new_lens.tolist() == [3 * 128]
    assert new_lens_list == [3 * 128]


def test_h2o_pruner_counts_partial_recent_block():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(heavy_blocks=1, recent_blocks=1)
    block_tables = torch.tensor([[20, 21, 22, 23, 24, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([4 * 128 + 17], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables[0, :2].tolist() == [20, 24]
    assert new_lens.tolist() == [128 + 17]
    assert new_lens_list == [128 + 17]


def test_h2o_pruner_leaves_short_sequences_unchanged():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(heavy_blocks=1, recent_blocks=1, min_seq_len=1024)
    block_tables = torch.tensor([[30, 31, 0, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([256], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert torch.equal(new_tables, block_tables)
    assert torch.equal(new_lens, seq_lens)
    assert new_lens_list == [256]


def test_h2o_pruner_reuses_existing_seq_lens_list():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(heavy_blocks=1, recent_blocks=2)
    block_tables = torch.tensor([[10, 11, 12, 13, 14, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([5 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
        seq_lens_list=[5 * 128],
    )

    assert new_tables[0, :3].tolist() == [10, 13, 14]
    assert new_lens.tolist() == [3 * 128]
    assert new_lens_list == [3 * 128]


def test_h2o_pruner_compacts_batch_with_single_gather_width():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=1,
        recent_blocks=1,
        adaptive_budget=False,
    )
    block_tables = torch.tensor(
        [
            [10, 11, 12, 13, 14, 0],
            [20, 21, 22, 23, 24, 0],
        ],
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([5 * 128, 5 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0", "req-1"],
    )

    assert new_tables.shape == (2, 2)
    assert new_tables.tolist() == [[10, 14], [20, 24]]
    assert new_lens.tolist() == [2 * 128, 2 * 128]
    assert new_lens_list == [2 * 128, 2 * 128]


def test_h2o_pruner_skips_compaction_when_prune_ratio_is_too_small():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=3,
        recent_blocks=1,
        adaptive_budget=False,
        min_prune_ratio=0.5,
    )
    block_tables = torch.tensor([[60, 61, 62, 63, 64, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([5 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables is block_tables
    assert new_lens is seq_lens
    assert new_lens_list == [5 * 128]
    assert pruner._decode_steps["req-0"] == 1


def test_h2o_pruner_clusters_cold_start_history_blocks():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=6,
        recent_blocks=1,
        adaptive_budget=False,
        history_cluster_size=2,
    )
    block_tables = torch.arange(12, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([12 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables[0, :7].tolist() == [0, 2, 3, 6, 7, 9, 11]
    assert new_lens.tolist() == [7 * 128]
    assert new_lens_list == [7 * 128]


def test_h2o_pruner_keeps_full_context_for_decode_warmup_steps():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=1,
        recent_blocks=1,
        adaptive_budget=False,
        decode_full_attention_steps=2,
    )
    block_tables = torch.tensor([[50, 51, 52, 53, 54, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([5 * 128], dtype=torch.int32)

    first_tables, first_lens, first_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )
    second_tables, second_lens, second_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )
    third_tables, third_lens, third_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert torch.equal(first_tables, block_tables)
    assert torch.equal(first_lens, seq_lens)
    assert first_lens_list == [5 * 128]
    assert torch.equal(second_tables, block_tables)
    assert torch.equal(second_lens, seq_lens)
    assert second_lens_list == [5 * 128]
    assert third_tables[0, :2].tolist() == [50, 54]
    assert third_lens.tolist() == [2 * 128]
    assert third_lens_list == [2 * 128]


def test_h2o_pruner_always_keeps_current_block():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(heavy_blocks=1, recent_blocks=0)
    block_tables = torch.tensor([[40, 41, 42, 43]], dtype=torch.int32)
    seq_lens = torch.tensor([4 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables[0, :2].tolist() == [40, 43]
    assert new_lens.tolist() == [2 * 128]
    assert new_lens_list == [2 * 128]


def test_h2o_pruner_expands_small_fixed_budget_for_long_context():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(heavy_blocks=2, recent_blocks=2)
    block_tables = torch.arange(100, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([100 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables[0, :10].tolist() == [0, 7, 21, 35, 49, 63, 77, 91, 98, 99]
    assert new_lens.tolist() == [10 * 128]
    assert new_lens_list == [10 * 128]


def test_h2o_pruner_can_keep_max_blocks_strict_when_precision_lift_disabled():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=2,
        recent_blocks=2,
        max_blocks=6,
        adaptive_precision_ratio=0.0,
    )
    block_tables = torch.arange(100, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([100 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables[0, :6].tolist() == [0, 17, 49, 81, 98, 99]
    assert new_lens.tolist() == [6 * 128]
    assert new_lens_list == [6 * 128]


def test_h2o_pruner_keeps_full_context_under_precision_cap():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=16,
        recent_blocks=16,
        max_blocks=32,
        adaptive_min_keep_ratio=0.0,
    )
    block_tables = torch.arange(75, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([75 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert torch.equal(new_tables, block_tables)
    assert torch.equal(new_lens, seq_lens)
    assert new_lens_list == [75 * 128]


def test_h2o_pruner_uses_scores_for_historical_anchors():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=5,
        recent_blocks=2,
        adaptive_budget=False,
        anchor_ratio=1.0,
    )
    pruner._scores["req-0"] = [0.0] * 20
    for block in (4, 8, 12, 16):
        pruner._scores["req-0"][block] = 10.0

    block_tables = torch.arange(20, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([20 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables[0, :7].tolist() == [0, 4, 8, 12, 16, 18, 19]
    assert new_lens.tolist() == [7 * 128]
    assert new_lens_list == [7 * 128]


def test_h2o_pruner_clusters_score_guided_anchors():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=6,
        recent_blocks=1,
        adaptive_budget=False,
        anchor_ratio=1.0,
        history_cluster_size=2,
    )
    pruner._scores["req-0"] = [0.0] * 12
    for block in (4, 8):
        pruner._scores["req-0"][block] = 10.0

    block_tables = torch.arange(12, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([12 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables[0, :7].tolist() == [0, 2, 3, 4, 5, 8, 11]
    assert new_lens.tolist() == [7 * 128]
    assert new_lens_list == [7 * 128]


def test_h2o_pruner_explores_historical_blocks_with_score_signal():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=6,
        recent_blocks=1,
        adaptive_budget=False,
        anchor_ratio=0.0,
        score_explore_ratio=0.5,
        score_coverage_ratio=0.0,
        selection_refresh_interval=1,
    )
    pruner._scores["req-0"] = [0.0] * 12
    for block in (1, 2, 3, 4, 5):
        pruner._scores["req-0"][block] = 10.0

    block_tables = torch.arange(12, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([12 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables[0, :7].tolist() == [0, 1, 2, 3, 4, 7, 11]
    assert new_lens.tolist() == [7 * 128]
    assert new_lens_list == [7 * 128]

    new_tables, _, _ = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables[0, :7].tolist() == [0, 1, 2, 3, 5, 8, 11]


def test_h2o_pruner_reuses_selection_between_refreshes():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=6,
        recent_blocks=1,
        adaptive_budget=False,
        anchor_ratio=0.0,
        score_explore_ratio=0.5,
        score_coverage_ratio=0.0,
        selection_refresh_interval=4,
    )
    pruner._scores["req-0"] = [0.0] * 12
    for block in (1, 2, 3, 4, 5):
        pruner._scores["req-0"][block] = 10.0

    block_tables = torch.arange(12, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([12 * 128], dtype=torch.int32)

    first_tables, _, _ = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )
    second_tables, _, _ = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert first_tables[0, :7].tolist() == [0, 1, 2, 3, 4, 7, 11]
    assert second_tables[0, :7].tolist() == [0, 1, 2, 3, 4, 7, 11]

    pruner._decode_steps["req-0"] = 4
    refreshed_tables, _, _ = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert refreshed_tables[0, :7].tolist() == [0, 1, 2, 3, 5, 7, 11]


def test_h2o_pruner_skips_score_updates_on_cache_hit_by_default():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=6,
        recent_blocks=1,
        adaptive_budget=False,
        anchor_ratio=0.0,
        score_explore_ratio=0.5,
        score_coverage_ratio=0.0,
        selection_refresh_interval=4,
    )
    block_tables = torch.arange(12, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([12 * 128], dtype=torch.int32)

    pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )
    scores_after_first = list(pruner._scores["req-0"])
    pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert pruner._scores["req-0"] == scores_after_first
    assert pruner._decode_steps["req-0"] == 2


def test_h2o_pruner_keeps_coverage_blocks_with_score_signal():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=8,
        recent_blocks=1,
        adaptive_budget=False,
        anchor_ratio=0.25,
        score_explore_ratio=0.0,
        score_coverage_ratio=0.5,
    )
    pruner._scores["req-0"] = [0.0] * 16
    for block in (1, 2, 3, 4, 5):
        pruner._scores["req-0"][block] = 10.0

    block_tables = torch.arange(16, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([16 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables[0, :9].tolist() == [0, 1, 2, 4, 6, 9, 11, 13, 15]
    assert new_lens.tolist() == [9 * 128]
    assert new_lens_list == [9 * 128]


def test_h2o_pruner_tapers_decode_budget_for_later_steps():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=16,
        recent_blocks=16,
        max_blocks=32,
        adaptive_min_keep_ratio=0.0,
        adaptive_precision_ratio=0.65,
        adaptive_precision_max_blocks=96,
        decode_budget_fast_ratio=0.45,
        decode_budget_taper_steps=128,
        decode_budget_taper_start_step=0,
    )
    pruner._decode_steps["req-0"] = 128
    block_tables = torch.arange(160, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([160 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables[0, :32].tolist()[-16:] == list(range(144, 160))
    assert new_lens.tolist() == [32 * 128]
    assert new_lens_list == [32 * 128]


def test_h2o_pruner_tapers_decode_budget_to_explicit_fast_blocks():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=48,
        recent_blocks=16,
        max_blocks=32,
        adaptive_min_keep_ratio=0.0,
        adaptive_precision_ratio=0.8,
        adaptive_precision_max_blocks=64,
        decode_budget_fast_blocks=32,
        decode_budget_fast_ratio=0.0,
        decode_budget_taper_steps=128,
        decode_budget_taper_start_step=0,
        selection_refresh_interval=16,
    )
    pruner._decode_steps["req-0"] = 128
    block_tables = torch.arange(80, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([80 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables.shape == (1, 64)
    assert new_tables[0, :4].tolist() == [0, 3, 7, 11]
    assert new_tables[0, :32].tolist()[-16:] == list(range(64, 80))
    assert new_tables[0, 32:].tolist() == [0] * 32
    assert new_lens.tolist() == [32 * 128]
    assert new_lens_list == [32 * 128]


def test_h2o_pruner_reuses_compact_metadata_indices_for_cached_selection():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=48,
        recent_blocks=16,
        max_blocks=32,
        adaptive_min_keep_ratio=0.0,
        adaptive_precision_ratio=0.8,
        adaptive_precision_max_blocks=64,
        decode_budget_fast_blocks=32,
        decode_budget_fast_ratio=0.0,
        decode_budget_taper_steps=128,
        decode_budget_taper_start_step=0,
        selection_refresh_interval=16,
    )
    pruner._decode_steps["req-0"] = 128
    block_tables = torch.arange(80, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([80 * 128], dtype=torch.int32)

    pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )
    assert pruner._compact_metadata_cache is not None
    cached_indices = pruner._compact_metadata_cache[1]

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert pruner._compact_metadata_cache[1] is cached_indices
    assert new_tables.shape == (1, 64)
    assert new_lens.tolist() == [32 * 128]
    assert new_lens_list == [32 * 128]


def test_h2o_pruner_recent_blocks_build_stronger_score_proxy():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=2,
        recent_blocks=3,
        adaptive_budget=False,
        sink_blocks=1,
    )
    block_tables = torch.arange(10, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([10 * 128], dtype=torch.int32)

    pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    scores = pruner._scores["req-0"]
    assert scores[0] > scores[2]
    assert scores[9] > scores[7] > scores[2]
