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
| `score_decay` | float | `1.0` | Decay for the lightweight retained-block score proxy. Must be in `(0, 1]`. |
| `adaptive_budget` | bool | `True` | When fixed block budgets are used, raise very small long-context budgets to `adaptive_min_keep_ratio`; when `max_blocks` is also set, `adaptive_precision_ratio` can further lift the selected-block target. |
| `adaptive_min_keep_ratio` | float | `0.1` | Minimum selected-block ratio for fixed `heavy_blocks`/`recent_blocks` budgets. Set to `0` to disable this minimum-ratio lift. |
| `adaptive_precision_ratio` | float | `0.6` | With fixed block budgets and `max_blocks`, lift the selected-block target to this ratio when the base cap would over-prune medium-long contexts. Set to `0` to make `max_blocks` a strict hard cap. |
| `adaptive_precision_max_blocks` | int | `96` | Upper bound for the `adaptive_precision_ratio` lift. If the current context has no more blocks than this value, H2O keeps the full context and skips Python-side pruning to protect accuracy and avoid overhead on medium contexts. Set to `None` to allow the ratio-based lift without this full-context guard. |
| `sink_blocks` | int | `1` | Number of initial blocks reserved from the heavy budget for system prompts and attention sinks. |
| `anchor_ratio` | float | `0.25` | Fraction of the remaining heavy budget reserved for score-guided historical anchor blocks when score signal exists. Cold starts use the whole remaining heavy budget as evenly spaced anchors. |
| `score_explore_ratio` | float | `0.2` | Fraction of the remaining heavy budget reserved for rotating historical exploration when score signal exists. This reduces retained-block score lock-in without increasing the selected-block count. |
| `score_coverage_ratio` | float | `0.35` | Fraction of the remaining heavy budget reserved for stable evenly spaced historical coverage when score signal exists. This keeps middle and late context represented while retaining the same selected-block count. |
| `decode_budget_fast_ratio` | float | `0.45` | Target selected-block ratio after the decode budget taper. Set to `0` to disable tapering. When `max_blocks` is set, the taper target is capped by `max_blocks` so late decode can return to the acceleration-oriented budget. |
| `decode_budget_taper_steps` | int | `256` | Number of decode steps used to move from the initial precision-oriented block target toward `decode_budget_fast_ratio`. Set to `0` to disable tapering. |
| `decode_budget_taper_start_step` | int | `64` | Number of initial decode steps to keep the full precision-oriented block target before tapering starts. |
| `selection_refresh_interval` | int | `4` | Number of decode steps between score-guided historical block reselections when the selected-block budget is stable. Set to `1` to recompute every step. Budget or context-length changes still refresh immediately. |
| `debug_log` | bool | `False` | Whether to log H2O pruning summaries for debugging. Keep this disabled for performance tests. |
| `debug_interval` | int | `1` | Print one debug summary every N decode metadata builds when `debug_log` is enabled. |
| `debug_sample_requests` | int | `3` | Number of sampled requests to include in each debug summary. |

Example:

```python
{
    "h2o_config": {
        "enabled": True,
        "heavy_ratio": 0.1,
        "recent_ratio": 0.1,
        "min_seq_len": 2048,
        "adaptive_min_keep_ratio": 0.1,
        "adaptive_precision_ratio": 0.65,
        "adaptive_precision_max_blocks": 96,
        "sink_blocks": 1,
        "anchor_ratio": 0.35,
        "score_explore_ratio": 0.2,
        "score_coverage_ratio": 0.35,
        "decode_budget_fast_ratio": 0.45,
        "decode_budget_taper_steps": 256,
        "decode_budget_taper_start_step": 64,
        "selection_refresh_interval": 4,
        "debug_log": False,
        "debug_interval": 50
    }
}
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
