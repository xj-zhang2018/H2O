# vllm-ascend H2O 服务化启动指南

## 启动方式

H2O 在 vllm-ascend 上不是独立服务，而是作为 **vLLM 推理引擎的注意力层优化** 集成，通过 `--additional-config` 机制启用。

### Online 模式（OpenAI兼容 API 服务）

```bash
vllm serve <model_path> \
  --additional-config='{"h2o_config":{"enabled":true,"heavy_ratio":0.1,"recent_ratio":0.1}}'
```

### Offline 模式（批量推理）

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-8B",
    additional_config={
        "h2o_config": {
            "enabled": True,
            "heavy_ratio": 0.1,
            "recent_ratio": 0.1,
        }
    }
)
```

## 生产级延迟优化启动命令

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_USE_V1=1 \
vllm serve /path/to/model \
  --served-model-name h2o-model \
  --tensor-parallel-size 8 \
  --max-model-len 12288 \
  --max-num-seqs 32 \
  --block-size 128 \
  --additional-config='{"h2o_config":{"enabled":true,"heavy_blocks":16,"recent_blocks":16,"max_blocks":32,"min_seq_len":4096,"adaptive_min_keep_ratio":0.0,"adaptive_precision_ratio":0.0,"adaptive_precision_max_blocks":null,"min_prune_ratio":0.50,"history_cluster_size":2,"sink_blocks":4,"anchor_ratio":0.30,"score_explore_ratio":0.15,"score_coverage_ratio":0.35,"decode_full_attention_steps":1,"decode_budget_fast_blocks":32,"decode_budget_fast_ratio":0.0,"decode_budget_taper_steps":0,"decode_budget_taper_start_step":0,"selection_refresh_interval":32,"score_update_on_cache_hit":false,"debug_log":false}}'
```

## 通用示例

```bash
vllm serve Qwen/Qwen3-8B \
  --additional-config='{"h2o_config":{"enabled":true,"heavy_ratio":0.1,"recent_ratio":0.1,"min_seq_len":2048,"adaptive_min_keep_ratio":0.1,"adaptive_precision_ratio":0.65,"adaptive_precision_max_blocks":96,"sink_blocks":1,"anchor_ratio":0.35,"score_explore_ratio":0.2,"score_coverage_ratio":0.35,"min_prune_ratio":0.0,"history_cluster_size":1,"decode_full_attention_steps":8,"decode_budget_fast_blocks":null,"decode_budget_fast_ratio":0.45,"decode_budget_taper_steps":256,"decode_budget_taper_start_step":64,"selection_refresh_interval":4,"score_update_on_cache_hit":false,"debug_log":false,"debug_interval":50}}'
```

## 配置流程

1. CLI `--additional-config` 解析 JSON → `vllm_config.additional_config`
2. `NPUPlatform.check_and_update_config()` → `init_ascend_config(vllm_config)`
3. `AscendConfig.__init__` → `H2OConfig(additional_config["h2o_config"])`
4. `AscendAttentionMetadataBuilder` → `H2OBlockPruner()` 实例化
5. 每次 decode → `_maybe_apply_h2o()` → 检查 `h2o_config.enabled` → `h2o_pruner.apply()`

## 关键配置参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `enabled` | false | 主开关 |
| `heavy_ratio` | 0.1 | Heavy hitter 预算比例 |
| `recent_ratio` | 0.1 | Recent 预算比例 |
| `heavy_blocks` | null | 固定 heavy block 数（覆盖 ratio） |
| `recent_blocks` | null | 固定 recent block 数（覆盖 ratio） |
| `max_blocks` | null | 每请求最大选中 block 数 |
| `min_seq_len` | 0 | 最小序列长度（低于此不裁剪） |
| `score_decay` | 1.0 | 分数衰减因子 |
| `sink_blocks` | 1 | 保留的初始 block 数（系统提示） |
| `anchor_ratio` | 0.25 | 锚点选择比例 |
| `selection_refresh_interval` | 4 | 重新选择间隔步数 |
| `decode_full_attention_steps` | 0 | 初始全量注意力步数 |

## 注意事项

- H2O 仅在 **decode-only** 阶段生效，prefill 不受影响
- `VLLM_USE_V1=1` 需要设置（v0.9.2+ 默认启用）
- Sliding-window 和 ALiBi 模型自动禁用 H2O
- `--block-size 128` 是 Ascend 平台的固定 block 大小