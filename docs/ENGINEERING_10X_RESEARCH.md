# Engineering investigation: can this MLX port become another 10× faster?

Date: 2026-09-19. Machine: Apple M3 Max, 40 GPU cores, 128 GiB unified memory,
macOS 27.2, MLX / MLX Metal 0.32.2, FP16 inference. This report contains actual
local experiments, including a hand-written Metal kernel. It does **not** change
the production runtime or publish quantized weights.

**The tested engineering changes do not deliver 10×.** Interleaved measurements
support modest, shape-dependent improvements from compilation and pruning unused
outputs of the final decision-head layer. Selected cases improved by approximately
3–8% using paired per-round medians. Some larger-batch intervals include no
improvement. A custom exact-erf GELU/gate kernel was numerically successful but did
not provide a consistent additional end-to-end benefit over MLX compilation.
Naive 8-bit and 4-bit backbone quantization reduced storage, failed to accelerate
the larger pilot workloads, and changed predictions or calibrated probabilities.

The mathematical limits and approximation tradeoffs are examined separately in
[MATH_10X_RESEARCH.md](MATH_10X_RESEARCH.md). The original implementation review is
in [PERFORMANCE_RESEARCH.md](PERFORMANCE_RESEARCH.md); the released checkpoint
benchmark remains [BENCHMARKS.md](../BENCHMARKS.md).

## Experimental controls and limits

All research GPU work ran serially. Other agent work used CPU/filesystem/network
only. The machine was on AC power, with no `pmset` thermal/performance warning
recorded and no swap use reported during the experiment. Normal desktop activity
continued. This is not a controlled thermal chamber or an otherwise idle
dedicated benchmark machine.

The first screening runs executed each candidate in a fresh process, with 4–5
warmups and 12–16 samples. They revealed substantial run-to-run drift. For example,
the English single-question pilot suggested a 1.24× compile improvement, whereas
the subsequent interleaved experiment found only about 1.03×. The sequential pilot
latencies are therefore screening evidence, not the primary causal speedup claim.

The confirmation script [paired.py](../experiments/engineering/paired.py):

- Rotates candidate order within each round and uses the same inputs for every
  candidate in that round.
- Changes actual state text between rounds. It generates up to 16 state variants
  and retains variants with the same tensor shape; multilingual short cases have
  10 such variants, while the other reported cases have 16.
- Uses distinct natural-language questions, including 50 different instructions
  for the largest short workload. It checks input hashes and does not cache
  answers, deduplicate questions, or reuse contextual encoder states.
- Evaluates results and synchronizes the GPU before stopping each timer. It
  measures both prepared forward calls and the public prediction path, including
  tokenization and output formatting. Model loading is excluded.
- Runs 32 measured rounds for the English head/compile experiment and 16 for the
  multilingual and custom-Metal experiments, after warmup. Each candidate sees
  the same round count and input sequence.

These inputs differ from the published baseline fixtures. The comparisons below
are **within the research experiment**, not before/after comparisons obtained by
dividing unrelated tables. The research batch limit is 64, while the released API
defaults to 16. Repeated state variants are intentional repeat measurements; there
is no result cache.

[analyze.py](../experiments/engineering/analyze.py) computes per-round
`eager_time / candidate_time` ratios and exploratory percentile bootstrap
intervals for their median, using 2,000 resamples of round indices. Those intervals
do not account for every source of operating-system noise or serial correlation
and are not a substitute for multi-session replication. A ratio of independently
computed p50 values can differ from the median paired ratio.

Raw JSON includes all timings, input hashes, environment metadata, parity
metrics, and the source fingerprint recorded at measurement time. The experiment
scripts were subsequently formatted and extended with disjoint optional
candidates; earlier fingerprints describe those earlier script versions.

## Compilation and exact final-head pruning

Four paths were compared:

1. **Eager:** the released FP16 `DecisionModel`.
2. **Compiled:** `mx.compile` around the loaded, evaluated, frozen model, using
   normal shape specialization.
3. **Selected Q + compiled:** in the last head layer, preserve full-length QKV
   projection and K/V, but issue only CLS/option-marker attention queries. Run the
   output projection and FFN only on these selected outputs.
4. **Full attention + selected outputs + compiled:** preserve the original
   full-length QKV and SDPA call, then gather CLS/option outputs before the output
   projection and FFN. This retains the original attention kernel shape while
   removing most unused final-head dense work.

Both pruning prototypes preserve the model's mathematical dependencies. They
still compute **all QKV projections**; they do not realize the additional Q-only
projection savings in the mathematical upper-bound calculation. Changing GEMM and
SDPA shapes can change floating-point rounding. Neither prototype is a decoder
cache, an early exit, or an approximation that drops earlier transformer layers.

End-to-end p50 latency, milliseconds:

| Model / request | B × L | Eager | Compiled | Selected Q + compiled | Full attention + selected outputs + compiled |
| --- | ---: | ---: | ---: | ---: | ---: |
| English short 1 | 1 × 78 | 16.628 | 16.185 | 15.925 | 15.636 |
| English short 16 | 16 × 82 | 116.920 | 113.700 | 112.009 | 110.217 |
| English long 1 | 1 × 512 | 53.921 | 53.078 | 52.301 | 52.121 |
| English long 8 | 8 × 512 | 531.166 | 518.428 | 504.135 | 488.980 |
| English short 50 | 50 × 82 | 456.333 | 439.013 | 445.223 | 438.293 |
| Multilingual short 1 | 1 × 80 | 8.050 | 7.570 | 7.438 | 7.388 |
| Multilingual short 16 | 16 × 83 | 44.351 | 43.830 | 42.281 | 42.968 |
| Multilingual long 1 | 1 × 1024 | 41.964 | 42.017 | 40.492 | 41.120 |
| Multilingual long 8 | 8 × 1024 | 326.327 | 323.053 | 327.842 | 319.010 |

Sources: [English paired data](../experiments/engineering/laya-paired.json) and
[multilingual paired data](../experiments/engineering/laya-multilingual-paired.json).

For the full-attention/selected-output path, paired median speedup and exploratory
95% intervals include:

| Request | Median paired speedup | Bootstrap interval |
| --- | ---: | ---: |
| English short 1 | 1.049× | 1.043–1.056× |
| English short 16 | 1.059× | 1.033–1.077× |
| English long 1 | 1.039× | 1.027–1.052× |
| English long 8 | 1.061× | 1.020–1.095× |
| English short 50 | 1.022× | 0.977–1.050× |
| Multilingual short 1 | 1.077× | 1.046–1.140× |
| Multilingual short 16 | 1.042× | 1.017–1.067× |
| Multilingual long 1 | 1.027× | 1.012–1.054× |
| Multilingual long 8 | 1.067× | 0.958–1.082× |

The English 50-question and multilingual long-batch intervals include 1. They do
not establish a repeatable improvement. The selected-Q path is somewhat better
for the multilingual short-16 and long-1 cases, but no one pruning path dominates
every shape. All candidate intervals, forward measurements, and raw per-round
ratios are in [paired_analysis.json](../experiments/engineering/paired_analysis.json).

Compilation exactly matched eager logits, action logits, and calibrated
probabilities on the 1,530 changed-input question comparisons across the two model
families in this head/compile experiment. Both pruning paths agreed on all 1,530
argmax decisions, with maximum calibrated probability difference **0.0001883**.
The full-attention pruning path also passed the separate 63-question fixture
suite for each model: 126/126 agreement, with maximum probability differences
4.31e-5 for English and 6.48e-6 for multilingual. These are regression checks,
not a claim of task accuracy on 1,530 independently labeled examples.

Whole-model and per-block compilation were both screened. The block experiment
also preserved all 63 English fixture outputs, but did not establish a material
advantage over whole-model compilation. Shape specialization must be bounded in
a service. The model uses Python shape-dependent reshapes and masks, so applying
`shapeless=True` indiscriminately is unsafe. The [official compile guide](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
documents shape specialization and state capture.

The first English whole-model candidate call took **2,166.7 ms**, followed by
about 12.75 ms warm forward p50 in that pilot; a new B16 shape first call took
272.4 ms. The JSON field is named `cold_forward`, but it means the **first
candidate call after eager reference inference**, not a fully cold application
or a freshly initialized Metal driver. Subsequent candidates reused previously
compiled Metal kernels, so their first-call times are not a controlled ranking
of cold-start cost. Compiled English short-1 active/peak MLX memory was about
803.6/918.6 MiB in the pilot; multilingual was about 614.1/676.9 MiB. These
allocator measurements do not include every host-side compiler allocation and
do not establish memory limits under unbounded shape churn. See
[English compile pilot](../experiments/engineering/laya-compiled-pilot.json) and
[multilingual compile pilot](../experiments/engineering/laya-multilingual-compiled-pilot.json).

## Selective quantization: useful storage savings, unsuitable as a speed claim

The prototype calls `nn.quantize` **after** loading the dense FP16 model. It
selects only `encoder.layers.*` linear modules, with affine group size 64, then
compiles the resulting model. Embeddings, norms, the decision head, scorer, and
action head remain FP16. This avoids casting packed integer weights through the
current dense loader and avoids the action head's non-divisible 1028/772 input
width. No quantized checkpoint format or loading contract is being shipped.
The [official MLX quantized layer implementation](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/quantized.py)
provides this selection mechanism.

| Model / encoder precision | Total tensor storage | Fixture agreement | Largest fixture probability change | Distinct workload agreement | Largest distinct-workload probability change |
| --- | ---: | ---: | ---: | ---: | ---: |
| English FP16 | 803.55 MiB | Reference | — | Reference | — |
| English 8-bit | 496.76 MiB | 62/63 | 0.0401 | 18/18 | 0.0312 |
| English 4-bit | 333.13 MiB | 50/63 | 0.3256 | 18/18 | 0.2224 |
| Multilingual FP16 | 613.99 MiB | Reference | — | Reference | — |
| Multilingual 8-bit | 515.38 MiB | 63/63 | 0.0133 | 26/26 | 0.0358 |
| Multilingual 4-bit | 462.79 MiB | 63/63 | 0.1268 | 19/26 | 0.8008 |

The multilingual 4-bit result illustrates why the small fixture suite alone is
insufficient: its 63 fixture argmaxes stayed the same, but 7 of 26 distinct
workload decisions changed. These are agreement measurements against FP16, not
ground-truth accuracy measurements. An absolute probability change of 0.8008 is
80.08 percentage points.

On English short-16 pilot inputs, FP16 eager/compiled end-to-end p50 was
91.26/87.94 ms; 8-bit/4-bit compiled was 96.66/93.20 ms. Short-1 quantization looked
somewhat faster in that screening run, while larger shapes did not. Multilingual
large-shape screening also failed to show a speed win, but its sequential runs
had substantial drift. These observations justify **rejecting an unqualified
speedup or release claim**, not assigning precise slowdown factors without
interleaved quantized replication. Further quantization work needs activation-aware
calibration or fine-tuning and a representative labeled quality suite.

Raw sources: [English 8-bit](../experiments/engineering/laya-q8-pilot.json),
[English 4-bit](../experiments/engineering/laya-q4-pilot.json),
[multilingual 8-bit](../experiments/engineering/laya-multilingual-q8-pilot.json),
[multilingual 4-bit](../experiments/engineering/laya-multilingual-q4-pilot.json).

## Hand-written Metal: exact GELU/gate fusion was implemented and tested

[kernels.py](../experiments/engineering/kernels.py) implements a real custom Metal
kernel that reads the two concatenated MLP branches, computes the same erf-based
GELU, multiplies by the gate, and writes a single output. It does not substitute
tanh-GELU or a sigmoid approximation. The kernel uses MLX v0.32.2's own erf and
expm1 helpers, preserving their licenses and notices in
[vendor/README.md](../experiments/engineering/vendor/README.md). It explicitly
supports FP16 only and uses safe Metal math mode. The [official custom-kernel guide](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
describes this API and its math-mode controls.

Across eight representative activation shapes, **27,958,016 randomly generated
FP16 output elements had exactly equal values to the original operation**. The
microbenchmark compares numerical equality, not the sign bit of zero. Full-model
changed-input tests also matched exactly: 474/474 question comparisons across
the two model families, plus both 63-question fixture suites, with zero logit,
action-logit, or calibrated-probability difference.

This correctness result did not translate into a consistent speed advantage over
MLX's fused compiled expression. For example, at 1,312 tokens and intermediate
width 2,624, per-call synchronized activation timing was 0.378 ms for eager
GELU-then-gate, 0.268 ms for `mx.compile`, and 0.280 ms for the custom kernel. At
8,192 tokens and width 1,152, the corresponding values were 0.846/0.764/0.714 ms.
These microbenchmarks include dispatch and synchronization overhead and are
screening probes; they are not measurements of isolated device execution time.
Full inputs, raw timings, and equality checks are in
[microbench.json](../experiments/engineering/microbench.json).

The custom kernel was then installed in **every encoder MLP** and measured in the
complete model with rotating candidate order and changing inputs:

| Model / request | Original compiled p50 | Metal + compiled p50 |
| --- | ---: | ---: |
| English short 1 | 23.795 ms | 23.837 ms |
| English short 16 | 142.716 ms | 139.355 ms |
| English long 1 | 68.241 ms | 68.982 ms |
| Multilingual short 1 | 7.557 ms | 7.437 ms |
| Multilingual short 16 | 49.683 ms | 50.301 ms |
| Multilingual long 1 | 48.906 ms | 51.032 ms |

The full custom-kernel paired runs use a second model instance with identical
weights so the unmodified and custom implementations coexist without mutation
or stale compiled captures. Their absolute timings must not be compared with
the earlier head-pruning run. The modest mixed results do not support publishing
the custom kernel as a general performance improvement. Sources:
[English Metal paired data](../experiments/engineering/laya-metal-paired.json) and
[multilingual Metal paired data](../experiments/engineering/laya-multilingual-metal-paired.json).

## Where custom engineering would be worth further investigation

The model already calls `mx.fast.scaled_dot_product_attention`, `mx.fast.rope`,
and optimized layer normalization. Its D64 boolean-mask SDPA path is fused; there
is no missing Flash Attention switch that explains a 10× gap. Local attention
still traverses dense key/value tiles. A real bidirectional window kernel could
skip those tiles while preserving inclusive distance <=64 and padding semantics,
but its whole-model arithmetic opportunity is small on short inputs and is
bounded on the published long shapes. The [existing source review](PERFORMANCE_RESEARCH.md#build-genuinely-local-attention-only-after-measuring-its-contribution)
and [math report](MATH_10X_RESEARCH.md) quantify this distinction.

Useful next projects, with their evidence requirements, are:

- **Long-input window attention:** specialize tile bounds for D64, the actual
  bidirectional window, and padded batches. Compare against fused dense SDPA at
  512/1024 tokens and then in the complete model. This kernel has not been built
  or benchmarked in this report.
- **Dense-kernel epilogues and scheduling:** investigate fusing the gated MLP
  epilogue into GEMM or improving short-M matrix scheduling. MLX already uses
  specialized Metal GEMM implementations, so replacing them requires a real
  dispatch/kernel profile and measured wins for the exact M/N/K shapes. The
  standalone activation result shows why another elementwise kernel alone is
  insufficient.
- **Length-aware batching and shared CPU preparation:** preserve exact input IDs
  while tokenizing shared state text once before constructing each question
  sequence, and avoid padding small items to unrelated long items. The multilingual
  long-8 pilot spent about 13.1 ms preparing inputs,
  versus hundreds of milliseconds end to end. Even eliminating that preparation
  entirely would not produce 10× on this workload. Queueing latency and unique
  inference count must be part of any batching claim.
- **A smaller jointly answering student:** if 10× is a product requirement,
  distill or redesign the model to remove most dense work or answer many fixed
  questions with one contextual encoding. This changes the learned model and
  needs representative labeled training/evaluation; it is not an exact port
  optimization. Reusing an arbitrary contextual state/KV across questions in the
  current bidirectional encoder is invalid.

Eight standalone FP16 encoder-input-projection GEMM probes achieved 0.66–11.55
TFLOP/s including per-call synchronization. The large English
`M=4096, N=5248, K=1024` probe achieved 11.55 TFLOP/s; the multilingual
`M=8192, N=2304, K=768` probe achieved 7.96 TFLOP/s. These are **observed throughput
values, not hardware peak specifications or upper bounds on full-graph
throughput**. Small-M measurements are particularly dominated by submission and
synchronization costs; a streamed graph amortizes them differently. They show
which shapes deserve profiling, not a proof that no better kernel can exist.
The 10× same-work throughput budgets in the math report remain theoretical
requirements rather than measured device capabilities.

## Reproduction and release decision

The scripts use the existing `.venv` and local pinned checkpoints. Run GPU
commands sequentially, never alongside the formal benchmark:

```bash
# Screening: repeat for eager, compiled, blocks, q8, q4, selected-compiled.
.venv/bin/python -m experiments.engineering.run_variants \
  --model laya --variant compiled --iterations 12 --warmup 4 --quality \
  --output experiments/engineering/reproduced-compiled.json

# Primary confirmation, including 50 genuinely different questions.
.venv/bin/python -m experiments.engineering.paired \
  --model laya --iterations 32 \
  --output experiments/engineering/reproduced-laya-paired.json
.venv/bin/python -m experiments.engineering.paired \
  --model laya-multilingual --iterations 16 --cases short1,short16,long1,long8 \
  --output experiments/engineering/reproduced-multilingual-paired.json

# Hand-written kernel microbench and complete-model comparison.
.venv/bin/python -m experiments.engineering.microbench
.venv/bin/python -m experiments.engineering.paired \
  --model laya --iterations 16 --cases short1,short16,long1 --metal \
  --output experiments/engineering/reproduced-metal-paired.json
.venv/bin/python -m experiments.engineering.run_variants \
  --model laya --variant metal-compiled --iterations 5 --warmup 3 \
  --cases short1 --quality --output experiments/engineering/reproduced-metal-quality.json

# CPU-only paired analysis.
.venv/bin/python -m experiments.engineering.analyze
```

All experimental Python files pass Ruff formatting and lint checks. The stable
runtime, original benchmark results, and published FP16 checkpoints remain the
release artifacts. Compilation and exact final-head pruning are credible
**optional future optimizations** after cold-shape/cache policy and broader
quality validation; the measured gains do not justify silently adding compilation
latency or a custom kernel to the default path. No 10× speedup, production-ready
quantized checkpoint, or measured local-window-kernel win is claimed.
