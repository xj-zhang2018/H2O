# other_code/ 目录：H2O 在华为 vllm-ascend 上的实现分析

## 概述

`other_code/vllm-ascend-releases-v0.18.0/` 是 H2O KV-cache 驱逐算法适配华为昇腾 NPU 平台（vllm-ascend）的实现代码。该实现将原始 H2O 的 token 级驱逐策略转化为 **block 级元数据优化**，在不修改物理 KV 缓存的前提下，通过压缩 block table 实现推理加速。

## 核心架构差异

| 维度 | 原始 H2O (`h2o_hf/`) | vllm-ascend H2O (`other_code/`) |
|---|---|---|
| 粒度 | token 级（逐 token 驱逐） | block 级（block_size=128 tokens） |
| 分数来源 | 真实注意力 softmax 概率 | 轻量级启发式代理分数 |
| 分数累积 | `hh_score = attn_weights.sum(0).sum(1)` 逐头逐 token | 逐请求逐 block 的 Python float 列表 |
| KV 驱逐方式 | 物理删除 KV 张量条目（boolean mask + view reshape） | 仅修改 block table 元数据，物理 KV 缓存不动 |
| 适用阶段 | prefill + decode | 仅 decode（`AscendAttentionState.DecodeOnly`） |
| 逐头评分 | 是：`(num_heads, seq_len)` | 否：单一分数 per block per request |
| 位置处理 | 需要调整 RoPE 位置 ID（streaming 变体中非常复杂） | 不需要：block table 压缩保留物理缓存，RoPE 不受影响 |
| 模型支持 | 通过 monkey-patching 特定注意力类（llama/opt/gpt_neox） | 任何使用 Ascend full-attention decode 的模型（架构无关） |

## 核心组件

### 1. H2OBlockPruner（`vllm_ascend/attention/h2o.py`）

核心类，1138 行，实现 block 级 KV 缓存选择策略。

**主要方法：**

| 方法 | 功能 |
|---|---|
| `apply()` | 主入口，接收 block_tables/seq_lens/block_size/config/request_ids，返回压缩后的 (block_tables, seq_lens, seq_lens_list) |
| `_select_blocks()` | 核心选择逻辑：多阶段 sink + recent + score-guided anchors + exploration + coverage + ranking |
| `_update_scores()` | 代理分数更新：含衰减(sink bonus +0.25, recency bonus +0.75*rank/recent_count, base +1.0) |
| `_resolve_budgets()` | 预算计算：ratio 或固定 block 计数，含自适应扩展 |
| `_build_compact_metadata()` | 构造压缩元数据：torch.gather 选取 block IDs，调整 seq_lens |

### 2. 选择算法（`_select_blocks()`）

多阶段选择策略，远比原始 H2O 的简单 topk 更复杂：

1. **Recent Blocks**：最后 `recent_blocks` 个 block 总是保留
2. **Sink Blocks**：前 `sink_blocks` 个 block（系统提示/attention sink），这是原始 H2O 没有的创新
3. **Heavy Blocks 选择**（冷启动 vs 有分数信号）：
   - **冷启动**（无分数）：`_clustered_evenly_spaced_blocks()` — 均匀分布 + 局部聚类
   - **有分数**（多策略混合）：
     - **Score-guided anchors** (`anchor_ratio=0.25`)：按分数桶区选最高分 block 作为锚点
     - **Cluster around anchors** (`history_cluster_size`)：锚点附近添加相邻 block 保证局部连贯
     - **Rotating exploration** (`score_explore_ratio=0.2`)：均匀分布 block，位置随 decode 步旋转，防止分数锁定
     - **Coverage blocks** (`score_coverage_ratio=0.35`)：稳定均匀分布，确保所有历史区域有代表
     - **Pure score ranking**：剩余预算按分数降序填充

### 3. 代理分数系统（`_update_scores()`）

由于 Ascend NPU 注意力内核不暴露 softmax 概率，使用启发式代理分数：

```
每一步：
1. 衰减所有分数：scores[j] *= score_decay (默认1.0=不衰减)
2. 对选中 block 增加分数：
   - Sink block: +1.0 + 0.25(sink bonus) = +1.25
   - Recent block: +1.0 + 0.75 * rank/recent_count (recency bonus)
   - 其他: +1.0 (base)
```

多次 decode 后，反复被选为 heavy hitter 的 block 累积高分；从未被选中的 block 在离开 recent 窗口后保持低分。

### 4. 自适应预算系统

29 个可配置参数，含多层自适应机制：

- **Decode warmup**：前 `decode_full_attention_steps` 步保持全量注意力
- **Budget taper**：从精度目标逐步过渡到加速目标 (`decode_budget_fast_ratio`, `decode_budget_taper_steps`)
- **Adaptive precision lift**：长上下文时扩展预算 (`adaptive_min_keep_ratio`, `adaptive_precision_ratio`)
- **Min prune ratio guard**：批量 prune 比率低于阈值时不压缩，避免元数据开销
- **Selection cache**：`selection_refresh_interval` 步间缓存选择结果

### 5. Block Table 压缩机制

关键创新：**不删除物理 KV 缓存**，只修改 block table：

```
原始 block_table: [物理block_0, block_1, block_2, ..., block_N]
压缩 block_table: [物理block_0, block_5, block_12, block_N-1]  (仅引用选中的 block)
seq_lens: 原始2048 → 压缩后 512 (4 blocks * 128 tokens/block)
```

Ascend 注意力内核只读取压缩 block table 引用的 block，实现注意力计算加速。

### 6. 自动禁用条件

H2O 在以下情况自动禁用：
- Sliding-window 模型（已有 KV 上下文限制）
- ALiBi 模型（位置偏置语义在跳过 block 时会改变）
- Cross-attention（编码器 KV 是独立的）
- Prefill/chunked-prefill 阶段（仅 decode 生效）

## 与原始 H2O 的对比总结

### 保留的核心思想
- Heavy hitter + Recent 双区保留策略
- 分数累积 + 裁剪机制
- 逐请求/逐层独立决策

### 关键创新
- Block 级粒度替代 token 级（适配 Ascend paged attention）
- 代理分数替代真实注意力（适配内核不暴露概率的限制）
- Block table 压缩替代物理 KV 删除（零数据移动，易回退）
- Sink blocks（系统提示保留）
- 多策略混合选择（anchors + clusters + exploration + coverage）
- 自适应预算 + decode taper（适配实际推理服务需求）
- Selection cache（减少 Python 侧开销）

### 局限性
- 物理 KV 缓存未被释放（内存节省仅在注意力计算层面）
- 代理分数比真实注意力概率精度低
- Block 粒度（128 tokens）比 token 级更粗
- 仅在 decode 阶段生效

## 关键文件清单

| 文件 | 路径 | 作用 |
|---|---|---|
| h2o.py | `vllm_ascend/attention/h2o.py` | 核心 H2OBlockPruner 实现 |
| attention_v1.py | `vllm_ascend/attention/attention_v1.py` | `_maybe_apply_h2o()` 集成入口 |
| ascend_config.py | `vllm_ascend/ascend_config.py` | H2OConfig 数据类 (29参数) |
| test_h2o.py | `tests/ut/attention/test_h2o.py` | 18个单元测试 |
| additional_config.md | `docs/source/user_guide/configuration/` | 用户配置文档 |
| install_flash_infer_attention_score_ops_a2.sh | `tools/` | Ascend 910B CANN内核替换（未来支持真实注意力分数） |
| install_flash_infer_attention_score_ops_a3.sh | `tools/` | Ascend 910C CANN内核替换 |