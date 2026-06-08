# H2O Project Guide

## Structure

Two independent sub-projects implementing the same H2O KV-cache eviction algorithm on different platforms:

- **h2o_flexgen/**: High-throughput inference via FlexGen offloading. Install with `pip install -e .` from this directory. Entrypoint: `python -m flexgen.flex_opt`. Key H2O args: `--hh-ratio`, `--hh-all`, `--hh-long-seq`.
- **h2o_hf/**: Benchmarking on HuggingFace Transformers. Entrypoints: `run_text_generation.py`, `run_lm_eval_harness.py`, `run_helm.py`, `run_streaming.py`, `run_summarization.py`. Key H2O args: `--enable_small_cache`, `--heavy_ratio`, `--recent_ratio`, `--hh_size`, `--recent_size`, `--enable_h2o_cache`.

`other_code/` is reference material (vllm-ascend) — not part of the H2O project.

## Core Architecture

H2O works by monkey-patching HuggingFace attention classes. Three separate patch sets exist, each with `modify_{llama,opt,gptneox}.py`:

- **utils_hh/**: Simulates H2O by masking the attention matrix (does not actually drop KV entries).
- **utils_real_drop/**: Implements real KV cache dropping. Also contains `stream.py` for streaming generation.
- **utils_lm_eval/**: Variant tailored for lm-eval-harness evaluation.

Each `convert_kvcache_*_heavy_recent()` function replaces the model's native attention module with the H2O variant at runtime. Supported model architectures: `llama`, `opt`, `gpt_neox` (selected via `--model_arch` or `--model-type`).

In `h2o_flexgen`, H2O is implemented via `cache_replace` / `acc_replace` in `flexgen/pytorch_backend.py`.

## Running Experiments

All experiments require a GPU and model weights (auto-downloaded from HuggingFace or loaded from a local `--cache-dir`).

**h2o_flexgen benchmarks** — run via suite definitions in `benchmark/h2o/h2o_suite.py`:
```bash
cd h2o_flexgen
python -m flexgen.flex_opt --gpu-batch-size 1 --overlap false --hh-ratio 0.1 --hh-all --model facebook/opt-6.7b
```

**h2o_hf benchmarks** — use the shell scripts in `scripts/`:
```bash
cd h2o_hf
bash scripts/streaming/eval.sh h2o       # streaming
bash scripts/summarization/eval.sh xsum 0 h2o 0 48 2000  # summarization
bash scripts/lm_eval/experiments.sh       # lm-eval-harness
```

lm-eval-harness has a 3-step pipeline: `generate_task_data.py` → `run_lm_eval_harness.py` → `evaluate_task_result.py`.

## HELM Setup Quirk

After installing `crfm-helm`, you must overwrite `toxicity_metrics.py` inside the installed package with the local version from `helm/src/helm/benchmark/metrics/toxicity_metrics.py`. The exact steps are documented in `h2o_hf/README.md`.

## Implementation Note

In `h2o_flexgen`, for n heavy hitters and n locals, the implementation actually preserves n-1 heavy hitters and n+1 locals after the first iteration (noted in `h2o_flexgen/README.md`).

## No Automated Tests

There is no test suite, CI, linter, or typechecker configured at the repo level. The `suite_test` list in `h2o_suite.py` is empty. Verification is done by running benchmark experiments on GPU hardware.