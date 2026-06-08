# Additional Configuration

Additional configuration is a mechanism provided by vLLM to allow plugins to control internal behavior by themselves. VLLM Ascend uses this mechanism to make the project more flexible.

## How to use

With either online mode or offline mode, users can use additional configuration. Take Qwen3 as an example:

**Online mode**:

```bash
vllm serve Qwen/Qwen3-8B --additional-config='{"config_key":"config_value"}'
```

**Offline mode**:

```python
from vllm import LLM

LLM(model="Qwen/Qwen3-8B", additional_config={"config_key":"config_value"})
```

### Configuration options

The following table lists additional configuration options available in vLLM Ascend:

| Name                                | Type | Default | Description                                                                                               |
|-------------------------------------|------|---------|-----------------------------------------------------------------------------------------------------------|
| `xlite_graph_config`                | dict | `{}`    | Configuration options for Xlite graph mode                                                                |
| `weight_prefetch_config`            | dict | `{}`    | Configuration options for weight prefetch                                                                 |
| `h2o_config`                        | dict | `{}`    | Configuration options for block-level H2O KV-cache pruning in decode attention                            |
| `finegrained_tp_config`             | dict | `{}`    | Configuration options for module tensor parallelism                                                       |
| `ascend_compilation_config`         | dict | `{}`    | Configuration options for ascend compilation                                                              |
| `eplb_config`                       | dict | `{}`    | Configuration options for eplb |
| `refresh`                           | bool | `false` | Whether to refresh global Ascend configuration content. This is usually used by rlhf or ut/e2e test case. |
| `dump_config_path`                  | str  | `None`  | Configuration file path for msprobe dump(eager mode).                                                     |
| `enable_async_exponential`          | bool | `False` | Whether to enable asynchronous exponential overlap. To enable asynchronous exponential, set this config to True.        |
| `enable_shared_expert_dp`           | bool | `False` | When the expert is shared in DP, it delivers better performance but consumes more memory. Currently only DeepSeek series models are supported. |
| `multistream_overlap_shared_expert` | bool | `False` | Whether to enable multi-stream shared expert. This option only takes effect on MoE models with shared experts. |
| `multistream_overlap_gate`          | bool | `False` | Whether to enable multi-stream overlap gate. This option only takes effect on MoE models with shared experts.  |
| `recompute_scheduler_enable`        | bool | `False` | Whether to enable the recompute scheduler. **Only valid in PD-disaggregated mode** (`kv_role` is `kv_producer` or `kv_consumer`). **Do not enable in PD-mixed mode** (no `kv_transfer_config`, or `kv_role` is `kv_both`); startup will fail with a clear error. |
| `enable_cpu_binding`                | bool | `True`  | Whether to enable CPU binding. Only takes effect on ARM CPUs; A3 uses the global-slicing CPU allocation strategy and other device types use the topo-affinity CPU allocation strategy. |
| `SLO_limits_for_dynamic_batch`      | int  | `-1`    | SLO limits for dynamic batch. This is new scheduler to support dynamic batch feature                            |
| `enable_npugraph_ex`                | bool | `False` | Whether to enable npugraph_ex graph mode.                                                                 |
| `pa_shape_list`                     | list | `[]`    | The custom shape list of page attention ops.                                                              |
| `enable_kv_nz`                      | bool | `False` | Whether to enable KV cache NZ layout. This option only takes effects on models using MLA (e.g., DeepSeek).                                      |
| `layer_sharding`                    | dict | `{}`    | Configuration options for Layer Sharding Linear. In PD-disaggregated deployments, it is supported only on P nodes with `kv_role="kv_producer"`. |
| `enable_sparse_c8`                  | bool | `False` | Whether to enable KV cache C8 in DSA models (e.g., DeepSeekV3.2 and GLM5). Not supported on A5 devices now |
| `enable_mc2_hierarchy_comm`         | bool | `False` | Enable dispatch/combine op inter-node communication by ROCE. |

The details of each configuration option are as follows:

**xlite_graph_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled` | bool | `False` | Whether to enable Xlite graph mode. Currently only Llama, Qwen dense series models, and Qwen3-VL are supported. |
| `full_mode` | bool | `False` | Whether to enable Xlite for both the prefill and decode stages. By default, Xlite is only enabled for the decode stage. |

**weight_prefetch_config**

| Name             | Type | Default                                                     | Description                        |
|------------------|------|-------------------------------------------------------------|------------------------------------|
| `enabled`        | bool | `False`                                                     | Whether to enable weight prefetch. |
| `prefetch_ratio` | dict | `{"attn": {"qkv": 1.0, "o": 1.0}, "moe": {"gate_up": 0.8}, "mlp": { "gate_up": 1.0,  "down": 1.0}}` | Prefetch ratio of each weight.     |

**h2o_config**

This option applies to full-attention decode. Sliding-window and ALiBi models keep the original attention metadata to avoid changing their positional/window semantics.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled` | bool | `False` | Whether to enable block-level H2O pruning during decode. |
| `heavy_ratio` | float | `0.1` | Historical heavy-hitter token budget ratio, rounded up to KV-cache blocks. |
| `recent_ratio` | float | `0.1` | Recent token budget ratio, rounded up to KV-cache blocks. |
| `heavy_blocks` | int | `None` | Optional fixed heavy-hitter block budget. Overrides `heavy_ratio` when set. |
| `recent_blocks` | int | `None` | Optional fixed recent block budget. Overrides `recent_ratio` when set. |
| `max_blocks` | int | `None` | Optional cap on selected blocks per request. When `adaptive_budget` and `adaptive_precision_ratio` are enabled, this is treated as the base cap and may be lifted or bypassed for accuracy-sensitive contexts. Recent blocks are kept first. |
| `min_seq_len` | int | `0` | Minimum sequence length before H2O pruning is applied. |
| `max_prune_seq_len` | int | `None` | Optional maximum sequence length for H2O pruning. Requests above this length keep the original full-attention metadata. Leave this unset when H2O must remain active for arbitrary long contexts. |
| `score_decay` | float | `1.0` | Decay for the lightweight retained-block score proxy. Must be in `(0, 1]`. |
| `adaptive_budget` | bool | `True` | When fixed block budgets are used, raise very small long-context budgets to `adaptive_min_keep_ratio`; when `max_blocks` is also set, `adaptive_precision_ratio` can further lift the selected-block target. |
| `adaptive_min_keep_ratio` | float | `0.1` | Minimum selected-block ratio for fixed `heavy_blocks`/`recent_blocks` budgets. Set to `0` to disable this minimum-ratio lift. |
| `adaptive_precision_ratio` | float | `0.6` | With fixed block budgets and `max_blocks`, lift the selected-block target to this ratio when the base cap would over-prune medium-long contexts. Set to `0` to make `max_blocks` a strict hard cap. |
| `adaptive_precision_max_blocks` | int | `96` | Upper bound for the `adaptive_precision_ratio` lift. If the current context has no more blocks than this value, H2O keeps the full context and skips Python-side pruning to protect accuracy and avoid overhead on medium contexts. When `max_blocks` and `adaptive_precision_ratio` are active, compact block-table metadata is padded to this width until an explicit `decode_budget_fast_blocks` target is reached. Set to `None` to allow the ratio-based lift without this full-context guard or static metadata padding. |
| `sink_blocks` | int | `1` | Number of initial blocks reserved from the heavy budget for system prompts and attention sinks. |
| `anchor_ratio` | float | `0.25` | Fraction of the remaining heavy budget reserved for score-guided historical anchor blocks when score signal exists. Cold starts use the whole remaining heavy budget as evenly spaced anchors. |
| `score_explore_ratio` | float | `0.2` | Fraction of the remaining heavy budget reserved for rotating historical exploration when score signal exists. This reduces retained-block score lock-in without increasing the selected-block count. |
| `score_coverage_ratio` | float | `0.35` | Fraction of the remaining heavy budget reserved for stable evenly spaced historical coverage when score signal exists. This keeps middle and late context represented while retaining the same selected-block count. |
| `min_prune_ratio` | float | `0.0` | Minimum batch-level pruned-block ratio required before compact block-table metadata is built. Set this above `0` to keep original metadata when the planned selected budget would not save enough attention work to offset Python/NPU metadata overhead; the guard runs before expensive block selection. |
| `min_metadata_prune_ratio` | float | `0.0` | Minimum batch-level block-table shape reduction required after padding compact metadata to one rectangular width. Set this to a small value such as `0.05` to protect continuous batching: if a warmup or full-context row would force every row back to full block-table width, H2O keeps original metadata instead of paying gather/update overhead with no effective metadata-shape savings. |
| `history_cluster_size` | int | `1` | Number of adjacent historical blocks to prefer around each selected historical anchor. Values greater than `1` improve local context continuity for accuracy-sensitive long prompts without increasing the selected-block budget. |
| `decode_full_attention_steps` | int | `0` | Number of initial decode metadata builds per request that keep the original full context before H2O pruning starts. This remains a manual warmup knob; automatic long-context mode has its own TTFT guard through `auto_tune_decode_warmup_steps`. |
| `decode_budget_fast_blocks` | int | `None` | Optional explicit selected-block target after the decode budget taper. When set with `decode_budget_fast_ratio=0`, it takes precedence so long-running decode can converge to a predictable acceleration-oriented block count. When set with a positive `decode_budget_fast_ratio`, it becomes the short-context floor and the ratio may lift longer contexts above this fixed count. Once the selected budget reaches the fixed target, compact metadata also uses this width instead of the precision padding width. |
| `decode_budget_fast_ratio` | float | `0.45` | Target selected-block ratio after the decode budget taper when `decode_budget_fast_blocks` is unset. When `decode_budget_fast_blocks` is also set, this ratio provides a length-aware minimum so one fixed fast block count does not over-compress longer prompts. Set to `0` to disable ratio-based tapering or lifting. When `max_blocks` is set and `decode_budget_fast_blocks` is unset, the taper target is capped by `max_blocks` so late decode can return to the acceleration-oriented budget. |
| `decode_budget_fast_max_blocks` | int | `None` | Optional upper bound for the length-scaled fast budget. Use this with `decode_budget_fast_blocks` and `decode_budget_fast_ratio` to keep H2O active for arbitrary long prompts without letting the selected block count grow linearly forever. When the selected budget is under this cap, compact block-table metadata is also padded to this width while `seq_lens` still limits the actual attended tokens, keeping decode graph/update shapes stable across prompt lengths. |
| `auto_tune` | bool | `True` | Enables the automatic long-context decode policy. When the active context has more blocks than `auto_tune_max_blocks`, H2O treats that value as a hard selected-block cap and prevents adaptive precision guards from silently keeping full-width metadata. Set this to `False` to preserve a fully manual H2O profile. |
| `auto_tune_max_blocks` | int | `64` | Maximum selected blocks and compact metadata width used by `auto_tune` for long contexts. For ratio-based automatic budgets, this also raises under-sized default budgets to the capped target, so a 20k-token prompt with 128-token KV blocks keeps up to 64 blocks instead of only the default 10% heavy + 10% recent budget. The final target is still length-aware through `decode_budget_fast_ratio`, but it is capped at this width so 10k, 20k, 32k, and longer prompts do not require separate per-length launch parameters. Set to `None` to keep `auto_tune` enabled without this cap. Explicit `heavy_blocks`/`recent_blocks` profiles are still respected. |
| `auto_tune_decode_warmup_steps` | int | `4` | Number of initial long-context decode metadata builds that keep the original full context before automatic H2O compaction starts. The default protects TTFT by leaving the first few decode metadata builds on the baseline full-metadata path, then applies compact metadata after warmup so long-output TPOT can still improve. Set to `0` only when first-token latency is excluded from the benchmark. |
| `decode_budget_taper_steps` | int | `256` | Number of decode steps used to move from the initial precision-oriented block target toward `decode_budget_fast_ratio`. Set to `0` to disable tapering. |
| `decode_budget_taper_start_step` | int | `64` | Number of initial decode steps to keep the full precision-oriented block target before tapering starts. |
| `selection_refresh_interval` | int | `4` | Number of decode steps between score-guided historical block reselections when the selected-block budget is stable. Set to `1` to recompute every step. Budget or context-length changes still refresh immediately. |
| `score_update_on_cache_hit` | bool | `False` | Whether to update retained-block scores when `selection_refresh_interval` reuses a cached selection. The default avoids duplicate Python-side score work and reduces score lock-in during cached decode steps. |
| `debug_log` | bool | `False` | Whether to log H2O pruning summaries for debugging. Keep this disabled for performance tests. |
| `debug_interval` | int | `1` | Print one debug summary every N decode metadata builds when `debug_log` is enabled. |
| `debug_sample_requests` | int | `3` | Number of sampled requests to include in each debug summary. |
| `debug_timing` | bool | `False` | Whether to log H2O timing summaries for metadata planning, block selection, compact metadata construction, and attention graph updates. Keep this disabled for normal performance tests. |
| `debug_timing_sync` | bool | `False` | Whether to synchronize the NPU before timing checkpoints. Enable only for short diagnostic runs because synchronization changes latency. |

Example:

```python
{
    "h2o_config": {
        "enabled": True,
        "heavy_ratio": 0.1,
        "recent_ratio": 0.1,
        "min_seq_len": 2048,
        "max_prune_seq_len": null,
        "adaptive_min_keep_ratio": 0.1,
        "adaptive_precision_ratio": 0.65,
        "adaptive_precision_max_blocks": 96,
        "sink_blocks": 1,
        "anchor_ratio": 0.35,
        "score_explore_ratio": 0.2,
        "score_coverage_ratio": 0.35,
        "min_prune_ratio": 0.0,
        "min_metadata_prune_ratio": 0.0,
        "history_cluster_size": 1,
        "decode_full_attention_steps": 0,
        "decode_budget_fast_blocks": null,
        "decode_budget_fast_ratio": 0.45,
        "decode_budget_fast_max_blocks": null,
        "auto_tune": True,
        "auto_tune_max_blocks": 64,
        "auto_tune_decode_warmup_steps": 4,
        "decode_budget_taper_steps": 256,
        "decode_budget_taper_start_step": 64,
        "selection_refresh_interval": 4,
        "score_update_on_cache_hit": False,
        "debug_log": False,
        "debug_interval": 50,
        "debug_timing": False,
        "debug_timing_sync": False
    }
}
```

For mixed 10k, 20k, 32k, and longer input / 1k output service benchmarks, the default automatic policy is the recommended starting point. It caps long-context decode metadata to a stable width, lifts the default ratio budget to that capped target for quality, keeps the first few decode metadata builds on the original full-metadata path to protect TTFT, applies compact metadata after warmup when the context is longer than the cap, and avoids having to retune `heavy_blocks`, `recent_blocks`, or `max_blocks` for each prompt length:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_USE_V1=1 \
vllm serve /path/to/model \
  --served-model-name h2o-model \
  --tensor-parallel-size 8 \
  --max-model-len 40960 \
  --max-num-seqs 32 \
  --block-size 128 \
  --additional-config='{"h2o_config":{"enabled":true,"min_seq_len":4096,"auto_tune":true,"auto_tune_max_blocks":64,"auto_tune_decode_warmup_steps":4,"selection_refresh_interval":128,"debug_log":false,"debug_timing":false,"debug_timing_sync":false}}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,12,16,32,64]}'
```

For mixed 10k, 20k, 32k, and longer input / 1k output, batch-size 32 service benchmarks, prefer the Ascend page-attention block size of 128 to reduce per-request block-table length before applying H2O. H2O is decode-only, so TTFT should stay close to the prefill baseline while TPOT and long-output E2E improve. Use the capped fast profile below when H2O should remain active for every prompt length without adding first-token setup overhead: it keeps graph-capture and the early real decode metadata full-width, then bounds runtime decode metadata after warmup, keeps a 32-block short-context floor, scales 20k and 30k prompts to smaller active KV windows, caps very long prompts at 64 selected blocks, pads compact block-table metadata to a stable 64-column width to avoid length-specific decode graph updates, keeps stronger sink and recent budgets for quality, and avoids `max_prune_seq_len` so 20k, 32k, and max-model-len-permitted 100k prompts still use compact H2O metadata after warmup. This profile is latency-first: on a 20k-token single request it can retain roughly 5k tokens after the warmup decode tokens, so use the conservative quality-validation profile below before judging long-context answer quality. The small metadata shape guard prevents 120-request continuous batches from mixing one full-context row with many pruned rows into a full-width gathered block table while preserving the intentional 64-column padding used by the 10k fast path.

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_USE_V1=1 \
vllm serve /path/to/model \
  --served-model-name h2o-model \
  --tensor-parallel-size 8 \
  --max-model-len 40960 \
  --max-num-seqs 32 \
  --block-size 128 \
  --additional-config='{"h2o_config":{"enabled":true,"auto_tune":true,"auto_tune_max_blocks":64,"auto_tune_decode_warmup_steps":4,"heavy_blocks":24,"recent_blocks":24,"max_blocks":32,"min_seq_len":4096,"adaptive_min_keep_ratio":0.0,"adaptive_precision_ratio":0.0,"adaptive_precision_max_blocks":null,"min_prune_ratio":0.50,"min_metadata_prune_ratio":0.05,"history_cluster_size":2,"sink_blocks":8,"anchor_ratio":0.25,"score_explore_ratio":0.25,"score_coverage_ratio":0.50,"decode_full_attention_steps":0,"decode_budget_fast_blocks":32,"decode_budget_fast_ratio":0.25,"decode_budget_fast_max_blocks":64,"decode_budget_taper_steps":0,"decode_budget_taper_start_step":0,"selection_refresh_interval":128,"score_update_on_cache_hit":false,"debug_log":false,"debug_timing":false,"debug_timing_sync":false}}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,12,16,32,64]}'
```

For a short 20k/32k latency diagnosis, keep the same service command but enable timing summaries and use a sparse interval:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_USE_V1=1 \
vllm serve /path/to/model \
  --served-model-name h2o-model \
  --tensor-parallel-size 8 \
  --max-model-len 40960 \
  --max-num-seqs 32 \
  --block-size 128 \
  --additional-config='{"h2o_config":{"enabled":true,"auto_tune":true,"auto_tune_max_blocks":64,"auto_tune_decode_warmup_steps":4,"heavy_blocks":24,"recent_blocks":24,"max_blocks":32,"min_seq_len":4096,"adaptive_min_keep_ratio":0.0,"adaptive_precision_ratio":0.0,"adaptive_precision_max_blocks":null,"min_prune_ratio":0.50,"min_metadata_prune_ratio":0.05,"history_cluster_size":2,"sink_blocks":8,"anchor_ratio":0.25,"score_explore_ratio":0.25,"score_coverage_ratio":0.50,"decode_full_attention_steps":0,"decode_budget_fast_blocks":32,"decode_budget_fast_ratio":0.25,"decode_budget_fast_max_blocks":64,"decode_budget_taper_steps":0,"decode_budget_taper_start_step":0,"selection_refresh_interval":128,"score_update_on_cache_hit":false,"debug_log":true,"debug_interval":50,"debug_timing":true,"debug_timing_sync":false}}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,12,16,32,64]}'
```

If asynchronous timing does not identify the bottleneck, repeat a much shorter run with `"debug_timing_sync":true` to measure synchronized NPU wall time. Do not compare that synchronized run directly against throughput benchmarks.

For single-request quality validation on long prompts, first compare with H2O disabled, then use a conservative H2O profile that keeps full attention for early decode tokens and raises the compact window before returning to the faster service profile:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_USE_V1=1 \
vllm serve /path/to/model \
  --served-model-name h2o-model \
  --tensor-parallel-size 8 \
  --max-model-len 40960 \
  --max-num-seqs 32 \
  --block-size 128 \
  --additional-config='{"h2o_config":{"enabled":true,"auto_tune":false,"heavy_blocks":48,"recent_blocks":48,"max_blocks":96,"min_seq_len":4096,"adaptive_min_keep_ratio":0.0,"adaptive_precision_ratio":0.50,"adaptive_precision_max_blocks":128,"min_prune_ratio":0.30,"min_metadata_prune_ratio":0.05,"history_cluster_size":2,"sink_blocks":16,"anchor_ratio":0.25,"score_explore_ratio":0.25,"score_coverage_ratio":0.50,"decode_full_attention_steps":64,"decode_budget_fast_blocks":96,"decode_budget_fast_ratio":0.50,"decode_budget_fast_max_blocks":128,"decode_budget_taper_steps":0,"decode_budget_taper_start_step":0,"selection_refresh_interval":32,"score_update_on_cache_hit":false,"debug_log":false,"debug_timing":false,"debug_timing_sync":false}}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,12,16,32,64]}'
```

**finegrained_tp_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `lmhead_tensor_parallel_size`    | int  | `0` | The custom tensor parallel size of lm_head.    |
| `oproj_tensor_parallel_size`     | int  | `0` | The custom tensor parallel size of o_proj.     |
| `embedding_tensor_parallel_size` | int  | `0` | The custom tensor parallel size of embedding. |
| `mlp_tensor_parallel_size`       | int  | `0` | The custom tensor parallel size of mlp.       |

**ascend_compilation_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enable_npugraph_ex`               | bool | `True` | Whether to enable npugraph_ex backend.                                                 |
| `enable_static_kernel` | bool | `False` | Whether to enable static kernel. Suitable for scenarios where shape changes are minimal and some time is available for static kernel compilation. |
| `fuse_norm_quant`  | bool | `True` | Whether to enable fuse_norm_quant pass. |
| `fuse_qknorm_rope` | bool | `True` | Whether to enable fuse_qknorm_rope pass. If Triton is not in the environment, set it to False. |
| `fuse_allreduce_rms` | bool | `False` | Whether to enable fuse_allreduce_rms pass. It's set to False because of conflict with SP. |
| `fuse_muls_add` | bool | `True` | Whether to enable fuse_muls_add pass.|

**eplb_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `dynamic_eplb`                   | bool| `False`| Whether to enable dynamic EPLB. |
| `expert_map_path`                | str | `None` | When using expert load balancing for an MoE model, an expert map path needs to be passed in.|
| `expert_heat_collection_interval`| int | `400`  | Forward iterations when EPLB begins. |
| `algorithm_execution_interval`   | int | `30`   | The forward iterations when the EPLB worker will finish CPU tasks. |
| `expert_map_record_path`         | str | `None` | Save the expert load calculation results to a new expert table in the specified directory.|
| `num_redundant_experts`          | int | `0`    | Specify redundant experts during initialization. |

### Example

An example of additional configuration is as follows:

```python
{
    "weight_prefetch_config": {
        "enabled": True,
        "prefetch_ratio": {
            "attn": {
                "qkv": 1.0,
                "o": 1.0,
            },
            "moe": {
                "gate_up": 0.8
            },
            "mlp": {
                "gate_up": 1.0,
                "down": 1.0
            }
        },
    },
    "finegrained_tp_config": {
        "lmhead_tensor_parallel_size": 8,
        "oproj_tensor_parallel_size": 8,
        "embedding_tensor_parallel_size": 8,
        "mlp_tensor_parallel_size": 8,
    },
    "enable_kv_nz": False,
    "multistream_overlap_shared_expert": True,
    "refresh": False
}
```
