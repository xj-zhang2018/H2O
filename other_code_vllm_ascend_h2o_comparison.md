# other_code/ 目录下的 H2O (vllm-ascend) 实现)对比分析

## 概述

 | `other_code/` 目录下的代码是 H2O KV-cache 驱逐算法适配华为昇腾 NPU 平台（vllm-ascend) 的实现代码，仅供参考，`other_code/` 不属于 H2O 项目的正式代码。 |
## 核心差异： Token级 vs Block级驱逐 | 2O 原始实现 (`h2o_hf/`) | vllm-ascend H2O (`other_code/`) | Token级驱逐 | 逐 token 物理删除 KV 条目， Block级驱逐 | 不删除 KV 条目， 只修改 block table 元数据 |
 | ----------| | ------------------- | |
 | 粒度 | Token级 (per-token) | Block级 (per-block, 128 tokens) |
 | 分数来源 | 真实注意力 softmax | 软代理分数 (衰减, sink, recency bonus) |
 | 分数累积 | `hh_score = attn_weights.sum(0).sum(1)`, shape `(heads, seq_len)` | 逐请求 per-block Python float 列表 |
 | 选择机制 | `torch.topk` on hh_score | 多策略: sink/recent/anchor/cluster/coverage/探索 |
 | KV 驱逐 | 物理删除 KV 张量 + reshape | 只修改 block table 指针 |
 | 适用阶段 | Prefill + Decode | Decode-only 模式（`AscendAttentionState.DecodeOnly`) |
 | 逐头评分 | `(num_kv_heads, seq_len)` per layer | `(num_heads, num_blocks)` per request |
 | 位置处理 | 需要处理 RoPE（streaming 模式） | 不需要，block table 保留物理缓存位置 |
 | 模型 | Monkey-patch 曍换注意力模块 | 使用 Ascend full-attention decode 架构 |
 | 配置参数 | 29个参数 (heavy_ratio, recent_ratio, decay, sink, anchor, coverage, taper 等) |
 | 自适应预算 | Warmup + Taper + Precision guards |
 | Block size | 1 token (GPU) | 128 tokens (Ascend NPU) |
 | 禁用条件 | Sliding window / ALiBi / Cross-attention / Chunked prefill |

## 关键文件说明

 | 文件 | 作用 |
 | --- | --- |
 | `h2o.py` | 核心 `H2OBlockPruner` 类，实现 block 级选择 |
 | `attention_v1.py` | `_maybe_enable_h2o()` 集成入口 |
 | `ascend_config.py` | `H2OConfig` 数据类 (29个配置参数) |
 | `test_h2o.py` | 18个单元测试 |
 | `configuration.md` | 配置文档 |
 | `install_*.sh` | Ascend CANN 内核安装脚本 (获取真实注意力分数) |
 
## 与原始 H2O (`h2o_hf/`) 的对比总结
 | 方面 | 原始 H2O | vllm-ascend H2O |
 | --- | --- | --- |
 | 核心思想 | 保留重要 token 的 KV | 保留重要 token 的 KV |
 | 实现粒度 | Token级 | Block级 (128 tokens/block) |
 | 分数来源 | 真实注意力 softmax | 软代理分数 (衰减+sink+recency bonus) |
 | 分数累积 | 逐 token sum | 逐 block Python float 列表 |
 | 选择策略 | 单一 topk | 多策略组合 (sink/recent/anchor/cluster/coverage/探索) |
 | KV 驱逐 | 物理删除 KV 张量 | 仅修改 block table 指针 |
 | 适用阶段 | Prefill + Decode | Decode-only 模式 |
 | 逐头评分 | per-layer per-request |
 | 位置处理 | 需要处理 RoPE | block table 保留物理缓存位置 |
 | 模型兼容 | Llama/OPT/GPT-NeoX (monkey-patch) | Ascend full-attention decode 架构 |
 | 配置 | 3-4 个核心参数 | 29 个精细参数 (自适应/衰减/taper) |
 | Block size | 1 token | 128 tokens/block |
 
 `other_code/` 中的实现更加成熟，有更精细的自适应机制（warmup/taper/precision guards）、更复杂的多策略组合选择算法，但这是为了适配华为昇腾 NPU 的 paged attention 架构所做的工程妥协，而非算法本身的改进。