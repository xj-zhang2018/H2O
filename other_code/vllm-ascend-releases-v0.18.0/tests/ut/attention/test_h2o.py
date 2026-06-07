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
    max_prune_seq_len: int | None = None
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
    decode_budget_fast_max_blocks: int | None = None
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


def test_h2o_pruner_leaves_long_sequences_above_max_prune_len_unchanged():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=1,
        recent_blocks=1,
        adaptive_budget=False,
        max_prune_seq_len=4 * 128,
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
    assert pruner._decode_steps == {}
    assert pruner._scores == {}
    assert pruner._selection_cache == {}


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
    assert "req-0" not in pruner._scores
    assert "req-0" not in pruner._selection_cache


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


def test_h2o_pruner_keeps_full_context_without_request_ids_for_warmup():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=1,
        recent_blocks=1,
        adaptive_budget=False,
        decode_full_attention_steps=1,
    )
    block_tables = torch.tensor([[60, 61, 62, 63, 64, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([5 * 128], dtype=torch.int32)

    for _ in range(2):
        new_tables, new_lens, new_lens_list = pruner.apply(
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=128,
            config=config,
            request_ids=None,
        )

        assert new_tables is block_tables
        assert new_lens is seq_lens
        assert new_lens_list == [5 * 128]

    assert pruner._decode_steps == {}
    assert pruner._scores == {}
    assert pruner._selection_cache == {}


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

    assert new_tables.shape == (1, 32)
    assert new_tables[0, :4].tolist() == [0, 3, 7, 11]
    assert new_tables[0, :32].tolist()[-16:] == list(range(64, 80))
    assert new_lens.tolist() == [32 * 128]
    assert new_lens_list == [32 * 128]


def test_h2o_pruner_prunes_first_decode_when_warmup_disabled():
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
        decode_budget_taper_start_step=16,
        decode_full_attention_steps=0,
        selection_refresh_interval=16,
    )
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
    assert new_tables[0, :64].tolist()[-16:] == list(range(64, 80))
    assert new_lens.tolist() == [64 * 128]
    assert new_lens_list == [64 * 128]


def test_h2o_pruner_skips_low_savings_taper_before_selection():
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
        decode_budget_taper_steps=64,
        decode_budget_taper_start_step=0,
        min_prune_ratio=0.30,
        selection_refresh_interval=16,
    )
    block_tables = torch.arange(80, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([80 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables is block_tables
    assert new_lens is seq_lens
    assert new_lens_list == [80 * 128]
    assert pruner._decode_steps["req-0"] == 1
    assert "req-0" not in pruner._scores
    assert "req-0" not in pruner._selection_cache


def test_h2o_pruner_strict_acceleration_profile_skips_first_then_prunes_fast_width():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=16,
        recent_blocks=16,
        max_blocks=32,
        adaptive_min_keep_ratio=0.0,
        adaptive_precision_ratio=0.0,
        adaptive_precision_max_blocks=None,
        min_prune_ratio=0.50,
        decode_full_attention_steps=1,
        decode_budget_fast_blocks=32,
        decode_budget_fast_ratio=0.0,
        decode_budget_taper_steps=0,
        decode_budget_taper_start_step=0,
        selection_refresh_interval=32,
    )
    block_tables = torch.arange(80, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([80 * 128], dtype=torch.int32)

    first_tables, first_lens, first_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert first_tables is block_tables
    assert first_lens is seq_lens
    assert first_lens_list == [80 * 128]
    assert pruner._decode_steps["req-0"] == 1

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables.shape == (1, 32)
    assert new_tables[0, :32].tolist()[-16:] == list(range(64, 80))
    assert new_lens.tolist() == [32 * 128]
    assert new_lens_list == [32 * 128]


def test_h2o_pruner_fast_block_floor_scales_for_long_contexts():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=16,
        recent_blocks=16,
        max_blocks=32,
        adaptive_min_keep_ratio=0.0,
        adaptive_precision_ratio=0.0,
        adaptive_precision_max_blocks=None,
        min_prune_ratio=0.50,
        decode_full_attention_steps=0,
        decode_budget_fast_blocks=32,
        decode_budget_fast_ratio=0.25,
        decode_budget_taper_steps=0,
        decode_budget_taper_start_step=0,
        selection_refresh_interval=32,
    )

    short_tables = torch.arange(80, dtype=torch.int32).unsqueeze(0)
    short_lens = torch.tensor([80 * 128], dtype=torch.int32)
    short_new_tables, short_new_lens, short_new_lens_list = pruner.apply(
        block_tables=short_tables,
        seq_lens=short_lens,
        block_size=128,
        config=config,
        request_ids=["short"],
    )

    assert short_new_tables.shape == (1, 32)
    assert short_new_tables[0, :32].tolist()[-16:] == list(range(64, 80))
    assert short_new_lens.tolist() == [32 * 128]
    assert short_new_lens_list == [32 * 128]

    long_tables = torch.arange(160, dtype=torch.int32).unsqueeze(0)
    long_lens = torch.tensor([160 * 128], dtype=torch.int32)
    long_new_tables, long_new_lens, long_new_lens_list = pruner.apply(
        block_tables=long_tables,
        seq_lens=long_lens,
        block_size=128,
        config=config,
        request_ids=["long"],
    )

    assert long_new_tables.shape == (1, 40)
    assert long_new_tables[0, :40].tolist()[-16:] == list(range(144, 160))
    assert long_new_lens.tolist() == [40 * 128]
    assert long_new_lens_list == [40 * 128]


def test_h2o_pruner_fast_capped_profile_keeps_long_context_active():
    pruner = H2OBlockPruner()
    warmup_steps = 8
    config = H2OConfigStub(
        heavy_blocks=24,
        recent_blocks=24,
        max_blocks=32,
        adaptive_min_keep_ratio=0.0,
        adaptive_precision_ratio=0.0,
        adaptive_precision_max_blocks=None,
        min_prune_ratio=0.50,
        decode_full_attention_steps=warmup_steps,
        decode_budget_fast_blocks=32,
        decode_budget_fast_ratio=0.25,
        decode_budget_fast_max_blocks=64,
        decode_budget_taper_steps=0,
        decode_budget_taper_start_step=0,
        selection_refresh_interval=128,
    )

    def apply_after_warmup(
        block_count: int,
        seq_len: int,
        request_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        block_tables = torch.arange(block_count, dtype=torch.int32).unsqueeze(0)
        seq_lens = torch.tensor([seq_len], dtype=torch.int32)
        for _ in range(warmup_steps):
            warm_tables, warm_lens, warm_lens_list = pruner.apply(
                block_tables=block_tables,
                seq_lens=seq_lens,
                block_size=128,
                config=config,
                request_ids=[request_id],
            )
            assert warm_tables is block_tables
            assert warm_lens is seq_lens
            assert warm_lens_list == [seq_len]

        new_tables, new_lens, new_lens_list = pruner.apply(
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=128,
            config=config,
            request_ids=[request_id],
        )
        return new_tables, new_lens, new_lens_list

    ten_k_new_tables, ten_k_new_lens, ten_k_new_lens_list = apply_after_warmup(
        block_count=80,
        seq_len=80 * 128,
        request_id="ten-k",
    )
    assert ten_k_new_tables.shape == (1, 64)
    assert ten_k_new_tables[0, :32].tolist()[-24:] == list(range(56, 80))
    assert ten_k_new_tables[0, 32:].tolist() == [0] * 32
    assert ten_k_new_lens.tolist() == [32 * 128]
    assert ten_k_new_lens_list == [32 * 128]

    twenty_k_new_tables, twenty_k_new_lens, twenty_k_new_lens_list = apply_after_warmup(
        block_count=160,
        seq_len=160 * 128,
        request_id="twenty-k",
    )
    assert twenty_k_new_tables.shape == (1, 64)
    assert twenty_k_new_tables[0, :40].tolist()[-24:] == list(range(136, 160))
    assert twenty_k_new_tables[0, 40:].tolist() == [0] * 24
    assert twenty_k_new_lens.tolist() == [40 * 128]
    assert twenty_k_new_lens_list == [40 * 128]

    thirty_k_new_tables, thirty_k_new_lens, thirty_k_new_lens_list = apply_after_warmup(
        block_count=240,
        seq_len=240 * 128,
        request_id="thirty-k",
    )
    assert thirty_k_new_tables.shape == (1, 64)
    assert thirty_k_new_tables[0, :60].tolist()[-24:] == list(range(216, 240))
    assert thirty_k_new_tables[0, 60:].tolist() == [0] * 4
    assert thirty_k_new_lens.tolist() == [60 * 128]
    assert thirty_k_new_lens_list == [60 * 128]

    hundred_k_new_tables, hundred_k_new_lens, hundred_k_new_lens_list = apply_after_warmup(
        block_count=782,
        seq_len=100000,
        request_id="hundred-k",
    )
    assert hundred_k_new_tables.shape == (1, 64)
    assert hundred_k_new_tables[0, :64].tolist()[-24:] == list(range(758, 782))
    assert hundred_k_new_lens.tolist() == [63 * 128 + 32]
    assert hundred_k_new_lens_list == [63 * 128 + 32]


def test_h2o_pruner_prunes_once_taper_has_enough_savings():
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
        decode_budget_taper_steps=64,
        decode_budget_taper_start_step=0,
        min_prune_ratio=0.30,
        selection_refresh_interval=16,
    )
    pruner._decode_steps["req-0"] = 64
    block_tables = torch.arange(80, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([80 * 128], dtype=torch.int32)

    new_tables, new_lens, new_lens_list = pruner.apply(
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=128,
        config=config,
        request_ids=["req-0"],
    )

    assert new_tables.shape == (1, 32)
    assert new_tables[0, :32].tolist()[-16:] == list(range(64, 80))
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
    assert new_tables.shape == (1, 32)
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
