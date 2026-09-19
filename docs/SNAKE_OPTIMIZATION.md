# Optimizing the actual Snake workload

The adopted opt-in path combines MLX compilation, 16-token length buckets and a bounded tokenized-question-prefix cache. It does **not** change weights, quantize the model, cache predictions, or reuse bidirectional encoder hidden states across questions.

In a paired complete-loop test, the shipped optimized path achieved **75.40 moves/second** over 2,400 moves, versus **70.82 moves/second** for eager inference in the same test: **1.065×**, or about 6.5%. Both had zero deaths, 2 safety interventions, and identical executed actions on 2,400/2,400 steps. Combined active MLX memory growth after warmup and cache clearing was **0 bytes**.

## Enable it

```bash
laya-snake --optimize
laya-snake --optimize --max-speed
```

The general API exposes the same opt-in controls:

```python
import laya_mlx as laya

agent = laya.load(
    "aac6fef/laya-multilingual-mlx",
    compile=True,
    pad_to_multiple=16,
    cache_prompts=True,
)
```

All three options default to disabled, preserving the existing eager behavior and benchmark configuration. Compilation specializes for input shape; first use and new shapes can incur compilation cost. Construct a new Agent after changing weights or module structure. Padding rounds sequence length up to the requested multiple without exceeding the configured context limit. Masks exclude padded tokens.

`cache_prompts=True` keeps at most 128 immutable `PreparedQuestion` prefixes per Agent, including marker positions. Cache keys include tokenizer identity, special tokens, question type, ordered rendered options, instructions and the prefix budget. The state is sanitized and tokenized once per `prepare` call, then independently concatenated with each question prefix. Question changes create or select the appropriate prefix. Every question still gets a full model forward.

## Shape and preparation ablation

This demo asks three questions per move: direction, safe-route estimate and food-reachability estimate. Thus its batch size is **3**, rather than 1. Across 32 sampled real recorded boards:

- Multilingual sequence lengths were **59, 61, 63 and 64**. A 16-token multiple puts all of them in a **64-token** bucket.
- English sequence lengths were **66, 68, 69 and 70**, mapping to an **80-token** bucket.
- Padding multilingual input from 64 to 96 adds 50% to token-wise work; it is not the small 93-to-96 adjustment suggested by the separate short-text API fixture.

Candidates ran in rotating order within each identical state, after visiting every measured shape once. The table contains synchronized `Agent.predict` latency including tokenization and output conversion, and excludes planner/UI work and initial shape warmup.

| Variant | Multilingual p50 / p95 (ms) | English p50 / p95 (ms) |
| --- | --- | --- |
| Eager | 9.12 / 10.21 | 21.83 / 26.73 |
| Prefix reuse only | 8.95 / 9.75 | 21.60 / 25.23 |
| Compiled, actual length | 8.67 / 9.66 | 21.27 / 25.99 |
| Compiled, padded to 96 | 10.92 / 11.72 | 25.66 / 29.88 |
| Compiled + prefix reuse | 8.66 / 9.21 | 21.03 / 23.97 |
| Compiled, workload bucket | 8.66 / 9.55 | 21.78 / 26.12 |
| Compiled + bucket + prefix reuse | 8.56 / 9.29 | 21.51 / 24.16 |

All seven candidates matched the eager proposed and executed directions on **32/32 boards per checkpoint**. The maximum difference in the displayed, four-decimal probabilities and estimates was **0** in these samples. This is finite-sample rounded-output agreement, not a claim of bit-identical internal floating-point tensors.

The ablation used bounded prefix preparation wrappers to screen designs. The complete-loop test below uses the actual shipped `compile`, `pad_to_multiple` and `cache_prompts` API implementation. Its correctness tests additionally compare prepared IDs and markers under state truncation, changing criteria, mask sanitation and cache eviction.

The shipped optimized path also passed the full real-checkpoint validation matrix: **63/63 selected-answer agreement for each of three checkpoints in FP32 and FP16 (378/378 total)**. Calibrated-probability errors remained within the existing tolerances. Each configuration passed 10 additional finite, deterministic repeated calls with **0 bytes** measured active-memory growth. [Optimized validation data](../benchmarks/results/validation-optimized.json). The original eager path's 100-repeat-per-configuration results remain in [the original benchmark report](../BENCHMARKS.md).

For English, compilation with the actual sequence length was better than forcing the larger bucket in this sample. The demo defaults to multilingual; general API users can leave `pad_to_multiple=None` while enabling compilation and prefix reuse.

## Complete-loop paired test

Four seeds, 600 moves each, with candidate order alternating per seed. Rendering includes truecolor Rich composition and ANSI serialization, and excludes the terminal emulator's painting. Each move performs a fresh prediction. Results are from one local paired run.

| Seed | Eager moves/s | Optimized moves/s | Score (both) | Action agreement |
| --- | --- | --- | --- | --- |
| 101 | 68.60 | 78.04 | 20 | 600 |
| 102 | 70.07 | 78.62 | 24 | 600 |
| 103 | 76.75 | 85.62 | 23 | 600 |
| 104 | 68.50 | 63.15 | 16 | 600 |

The optimized path was slower on one seed. Consequently, **6.5% is the combined improvement in this measured run**, rather than a guaranteed improvement for every episode or machine. The earlier broad speed sweep and this later paired test are different runs; their absolute rates must not be subtracted to claim a speedup. [Full loop data](../benchmarks/results/snake-optimized-paired.json).

## Choose the model using gameplay as well as latency

Both checkpoints ran **20 paired seeds × 300 moves**, with checkpoint order alternating each seed. Every episode used the same initial state, food RNG seed, compact feature descriptions and cycle shield. The horizon is fixed; these are scores after 300 moves, not complete games ending in death or a filled board. No terminal rendering was included in this model comparison.

| Checkpoint | Survived / episodes | Moves | Median / mean score | Inference p50 / p95 (ms) | Interventions |
| --- | --- | --- | --- | --- | --- |
| laya | 20 / 20 | 6000 | 7.0 / 6.9 | 23.15 / 28.21 | 0 |
| multilingual | 20 / 20 | 6000 | 10.0 / 9.9 | 9.38 / 14.38 | 2 |

Multilingual made more food progress and was faster on this workload, so it remains the default demo checkpoint. The result evaluates this feature-assisted policy, not general reasoning quality or an unassisted Snake model. [Every episode and inference](../benchmarks/results/snake-model-comparison.json).

## Reproduce

```bash
uv run --extra demo python -m experiments.snake_runtime \
  --output artifacts/snake/runtime-multilingual.json
uv run --extra demo python -m experiments.snake_runtime \
  --model models/hub/laya-mlx --bucket 80 \
  --output artifacts/snake/runtime-english.json
uv run --extra demo python -m benchmarks.snake_optimized \
  --output artifacts/snake/optimized-paired.json
uv run --extra demo python -m benchmarks.snake_models \
  --episodes 20 --steps 300 --output artifacts/snake/models.json
```

Run GPU measurements sequentially. Download the two local model directories first. The checked-in source recording supplies the exact sampled board states. Raw ablations: [multilingual](../benchmarks/results/snake-runtime-multilingual.json), [English](../benchmarks/results/snake-runtime-english.json), [initial 96-token pilot](../benchmarks/results/snake-runtime-96-pilot.json).

The earlier [compact versus detailed prompt comparison](../benchmarks/results/snake-prompt-comparison.json) alternated prompt order on 64 states and found 11.80 → 9.29 ms median after discarding the first 8 warmup iterations. Its original ad-hoc record did not store board snapshots, so it is supporting evidence rather than the primary reproducible ablation. `python -m benchmarks.snake_prompt` provides a reproducible version that stores states, seed, full decisions and method.

The implementation follows MLX's [official compilation guide](https://ml-explore.github.io/mlx/build/html/usage/compile.html): use a long-lived compiled callable and normal shape specialization. It does not use `shapeless=True` on shape-dependent Python model code. Current documentation was checked through the official site after Context7 CLI requests failed with network errors.
