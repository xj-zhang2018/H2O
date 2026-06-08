# H2O (Heavy-Hitter Oracle) KV-Cache Eviction Algorithm

## Core Principle

H2O maintains a **fixed-size KV cache** during autoregressive generation by preserving only two categories of tokens:

- **Heavy Hitters**: Tokens that accumulate the highest cumulative attention scores across all past query tokens. These are "important" tokens that many subsequent tokens attend to heavily.
- **Recent Tokens**: The most recently generated tokens, always retained because they provide local context for autoregressive generation.

The cache budget is defined as `cache_size = hh_size + recent_size`. When the KV cache length exceeds this budget, eviction decisions are made at every decoding step.

## Accumulated Attention Score

At each decoding step, the model computes attention weights `A_t` over all keys in the KV cache. After softmax normalization, the accumulated attention score for position `j` at step `t` is:

```
S_t[j] = S_{t-1}[j] + Σ_h A_t[h, j]
```

Where:
- `S_{t-1}[j]` is the previously accumulated score for position `j`
- `A_t[h, j]` is the attention weight that head `h` assigns to key position `j` at step `t`

For the prefill phase (processing the entire prompt at once):

```
S_prefill[j] = Σ_i Σ_h A[i, h, j]
```

Tokens with high accumulated scores are "heavy hitters" — they receive consistently high attention from many query tokens across many steps.

## Algorithm Flow

### Phase 1: Prefill

1. Compute standard self-attention over the full prompt sequence
2. Obtain attention weights `A` of shape `[batch, heads, prompt_len, prompt_len]`
3. Compute accumulated importance scores: `S[j] = Σ_i Σ_h A[i, h, j]` for each key position `j`
4. Define `cache_budget = heavy_budget + recent_budget`
5. If `prompt_len > cache_budget`:
   - Partition positions into:
     - Non-recent zone: `[0, prompt_len - recent_budget)`
     - Recent zone: `[prompt_len - recent_budget, prompt_len)`
   - From the non-recent zone, select top-`heavy_budget` positions by accumulated score
   - Keep the recent zone intact
   - Evict (or mask) all other positions
   - Prune accumulated scores to match the new cache size

### Phase 2: Decoding (per generated token)

1. Concatenate the new token's K, V to the existing cache
2. Compute attention over the cache: `A_t` of shape `[batch, heads, 1, cache_len]`
3. Accumulate: `S_t[j] = S_{t-1}[j] + Σ_h A_t[h, j]` for each existing position
4. If `cache_len > cache_budget`:
   - Partition into non-recent `[0, cache_len - recent_size)` and recent `[cache_len - recent_size, cache_len)`
   - Select top-`heavy_budget` from non-recent zone by `S_t` via `torch.topk`
   - Keep recent zone intact
   - Evict all other positions
   - Prune scores for evicted positions
5. Compute attention output (using full pre-eviction weights in real-drop variant)

## Three Implementation Variants

### 1. `utils_real_drop/` — Real KV Cache Eviction

Physically removes KV entries from the cache. Core class: `H2OKVCache_LayerWise`.

Key features:
- Each layer maintains its own cache and accumulated scores independently
- `evict_for_space()` method for pre-allocating space before new tokens arrive (used in streaming)
- Position rolling with re-applied RoPE embeddings for streaming mode
- Attention output uses full context before eviction

### 2. `utils_hh/` — Attention Masking Simulation

Does NOT actually drop KV entries. Instead applies binary mask to attention matrix to simulate the effect.

Key features:
- `previous_scores`: accumulated attention scores per head per position
- `attention_masks_next`: binary mask applied BEFORE softmax at the next step
- Mask has `kv_len + 1` columns (extra column for the next new token)
- Evicted positions get `-inf` in the mask, producing zero attention after softmax
- Budget ratios (`heavy_ratio`, `recent_ratio`) converted to absolute counts at initialization

### 3. `utils_lm_eval/` — Global Statistics Variant

Uses accumulated softmax attention scores in a single pass (global approach).

Key features:
- Single-shot prefill: sums full attention matrix over all query positions to get global importance score
- Recent tokens defined as a diagonal band: each query position sees `recent_budget` positions immediately before/after
- Heavy hitter positions visible to ALL query positions

### 4. FlexGen Implementation (`h2o_flexgen/`)

High-throughput offloading variant with fixed-size pre-allocated arrays.

Key features:
- Prefill: keeps `hh_k` heavy hitters + `hh_k - 1` recent tokens (total `2*hh_k - 1`)
- Decoding: incremental one-at-a-time eviction — evicts the "light hitter" (minimum accumulated score) and replaces with the new token
- Rolling recent window: oldest recent token gets replaced, and the least-important heavy hitter moves into that freed slot

## Key Design Properties

| Property | Description |
|---|---|
| Per-layer independence | Each attention layer maintains its own cache and scores. Different layers may preserve different heavy hitters |
| Score pruning | After eviction, scores for evicted positions are zeroed out or physically removed, preventing ghost scores from biasing future selections |
| Recent token protection | No matter how low a recent token's score is, it is never evicted until it ages out of the recent window |
| Full-context attention output | In real-drop variant, attention output is computed before eviction — the model always produces its best possible output |
| RoPE position rolling | In streaming mode, positions are reassigned as `[0, 1, ..., cache_size-1]` to maintain correct rotary embeddings |

## Supported Model Architectures

| Architecture | Real-drop | Masking | LM-eval | FlexGen |
|---|---|---|---|---|
| LLaMA | Yes (standard + streaming) | Yes | Yes | — |
| OPT | — | Yes | Yes | Yes |
| GPT-NeoX | — | Yes | Yes | — |

## Configurable Parameters

| Parameter | Default | Description |
|---|---|---|
| `--heavy_ratio` / `--hh_ratio` | 0.1 | Ratio of sequence for heavy hitters |
| `--recent_ratio` | 0.1 | Ratio of sequence for recent tokens |
| `--hh_size` | 1024 | Absolute count of heavy hitter slots |
| `--recent_size` | 1024 | Absolute count of recent token slots |
| `--enable_h2o_cache` | off | Enable H2O KV cache |
| `--model_arch` | llama | Architecture: llama/opt/gpt_neox |