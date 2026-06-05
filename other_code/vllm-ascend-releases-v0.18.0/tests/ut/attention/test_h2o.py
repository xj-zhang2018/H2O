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


def test_h2o_pruner_lifts_low_max_blocks_for_medium_long_context():
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

    assert new_tables[0, :45].tolist() == [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        31,
        33,
        35,
        37,
        39,
        41,
        43,
        45,
        47,
        49,
        51,
        53,
        55,
        57,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
    ]
    assert new_lens.tolist() == [45 * 128]
    assert new_lens_list == [45 * 128]


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


def test_h2o_pruner_explores_historical_blocks_with_score_signal():
    pruner = H2OBlockPruner()
    config = H2OConfigStub(
        heavy_blocks=6,
        recent_blocks=1,
        adaptive_budget=False,
        anchor_ratio=0.0,
        score_explore_ratio=0.5,
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
