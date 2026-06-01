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
