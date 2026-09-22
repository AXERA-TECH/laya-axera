# Local benchmarks

Measured on **Apple M3 Max**, 40-core GPU, **128 GiB unified memory**, macOS-27.2-arm64-64bit, Python 3.12.13.

These are local measurements of the native MLX port and the pinned upstream PyTorch runtime on the same Mac. They are not comparisons with the upstream T4 or third-party API figures.

## Method

- Each checkpoint/backend runs in a fresh process, sequentially, with 5 warmup iterations and 50 timed iterations per workload and timing mode.
- End-to-end timing includes prompt construction, tokenization, tensor construction, model execution, calibration, and result formatting. Model loading and downloads are excluded.
- Forward timing uses prepared device tensors and includes the encoder, decision head, scoring head and action head. GPU completion is synchronized for every sample; lazy MLX graph construction alone is never timed as inference.
- All backends receive identical state and question JSON. The report generator verifies matching input hashes, token totals and padded sequence lengths.
- The 5/10/50-question fixtures cycle three question templates. The released runtime evaluates every row and has no result cache or deduplication; these rows measure repeated-template batch throughput. Optimization studies should add distinct-question workloads, as detailed in docs/PERFORMANCE_RESEARCH.md.
- MLX uses batch_size=64 for these measurements so even 50 questions fit in one batch. The public runtime defaults to 16 to bound memory; changing batch size can change throughput.
- PyTorch MPS uses upstream's default FP32. MLX FP32 provides the same-precision comparison. MLX FP16 trades some numerical precision for speed and memory; its speedup includes that precision change.
- P50/P95 are percentiles of measured wall-clock latency. Throughput is questions / mean latency, not the inverse of P50. Raw JSON contains every timing sample.
- This is one development machine and one run of each configuration, with normal OS activity. No claim of cross-device performance or production endurance is made.

![Local latency comparison](benchmarks/latency.png)

## End-to-end short input latency

All values are milliseconds per request. P95 is shown after `/`.

| Checkpoint | Questions | Tokens / padded length | PyTorch MPS FP32 P50 / P95 | MLX FP32 P50 / P95 | MLX FP16 P50 / P95 | FP16 questions/s |
|---|---:|---:|---:|---:|---:|---:|
| laya | 1 | 93 / 93 | 22.70 / 25.47 | 18.55 / 20.25 | 17.75 / 21.45 | 54.9 |
| laya | 5 | 427 / 93 | 56.41 / 59.58 | 59.06 / 69.58 | 44.43 / 49.00 | 111.1 |
| laya | 10 | 855 / 93 | 95.77 / 124.46 | 103.76 / 116.21 | 80.82 / 92.04 | 122.5 |
| laya | 50 | 4237 / 93 | 489.54 / 515.80 | 462.36 / 504.59 | 347.24 / 372.44 | 143.3 |
| laya-multilingual | 1 | 91 / 91 | 13.60 / 14.57 | 10.73 / 19.78 | 10.91 / 19.48 | 81.7 |
| laya-multilingual | 5 | 432 / 91 | 27.82 / 28.98 | 21.94 / 23.72 | 19.28 / 20.81 | 268.5 |
| laya-multilingual | 10 | 862 / 91 | 43.87 / 45.70 | 37.47 / 40.42 | 32.92 / 39.56 | 295.3 |
| laya-multilingual | 50 | 4287 / 91 | 195.16 / 217.52 | 159.39 / 167.86 | 125.49 / 134.11 | 402.2 |
| laya-typed-decisions | 1 | 93 / 93 | 22.85 / 25.62 | 18.53 / 21.42 | 16.17 / 17.74 | 61.7 |
| laya-typed-decisions | 5 | 427 / 93 | 57.44 / 58.65 | 52.13 / 54.90 | 45.70 / 48.92 | 110.1 |
| laya-typed-decisions | 10 | 855 / 93 | 98.80 / 101.75 | 91.12 / 95.09 | 81.38 / 86.05 | 122.8 |
| laya-typed-decisions | 50 | 4237 / 93 | 489.12 / 514.73 | 410.98 / 452.81 | 325.76 / 335.83 | 153.2 |

## Full-context latency

Long input fills each checkpoint's configured limit, including question and option tokens.

| Checkpoint | Questions | Padded length | PyTorch MPS FP32 P50 | MLX FP32 P50 | MLX FP16 P50 |
|---|---:|---:|---:|---:|---:|
| laya | 1 | 512 | 64.09 | 64.79 | 49.84 |
| laya | 10 | 512 | 653.74 | 540.00 | 420.46 |
| laya-multilingual | 1 | 1024 | 54.39 | 51.47 | 43.50 |
| laya-multilingual | 10 | 1024 | 528.11 | 452.67 | 338.82 |
| laya-typed-decisions | 1 | 1024 | 118.64 | 115.59 | 99.95 |
| laya-typed-decisions | 10 | 1024 | 1258.13 | 1116.65 | 885.05 |

## Memory

MLX peak allocated memory includes model weights, inputs and intermediates; cache memory is recorded separately. MPS current tensor/driver allocations in the raw JSON are different metrics and are not presented as comparable peaks.

| Checkpoint | Parameters | FP16 weights (MiB) | FP16 peak: 1 short question (MiB) | FP16 peak: 10 full-context questions (MiB) |
|---|---:|---:|---:|---:|
| laya | 421,293,827 | 803.6 | 943.6 | 1833.0 |
| laya-multilingual | 321,908,995 | 614.0 | 687.6 | 1509.1 |
| laya-typed-decisions | 421,293,827 | 803.6 | 943.6 | 1643.7 |

## Numerical parity and stability

Real checkpoint validation covers 16 cases and 63 questions per checkpoint and precision: eight languages, empty and long states, conversation lists, mask-token literals, structured criteria, mixed question batches and 20 options. This measures fidelity to upstream, not correctness of every model answer.

| Checkpoint | Precision | Argmax agreement | Max calibrated probability error | Repeated calls | Active memory growth (bytes) |
|---|---|---:|---:|---:|---:|
| laya | float32 | 63/63 | 0.0000052 | 100 | 0 |
| laya | float16 | 63/63 | 0.0054443 | 100 | 0 |
| laya-multilingual | float32 | 63/63 | 0.0000012 | 100 | 0 |
| laya-multilingual | float16 | 63/63 | 0.0012887 | 100 | 0 |
| laya-typed-decisions | float32 | 63/63 | 0.0000027 | 100 | 0 |
| laya-typed-decisions | float16 | 63/63 | 0.0016849 | 100 | 0 |

Every repeated call checks finite logits and exactly repeatable public JSON for the same input and batch shape. Memory growth is measured after garbage collection and clearing the MLX cache. The per-case JSON also records raw action-logit error: action logits can have large magnitudes, so action softmax probabilities are checked separately. FP32 probability tolerance is 0.0001 and FP16 tolerance is 0.02; these thresholds were set in the validation script before measuring.

## Labeled task sample

AG News test split: 256 examples, equal class counts, seed 20260919. This is a small English classification sample; AG News appears in upstream's training mix, and the typed-decisions checkpoint targets different tasks. Source revision, sampled indices, gold labels and both backends' predictions are in `benchmarks/results/accuracy.json`. No sample text is redistributed.

| Checkpoint | Upstream MPS FP32 accuracy | MLX FP16 accuracy | Prediction agreement |
|---|---:|---:|---:|
| laya | 0.9570 | 0.9570 | 256/256 |
| laya-multilingual | 0.9453 | 0.9453 | 256/256 |
| laya-typed-decisions | 0.9648 | 0.9648 | 256/256 |

## Versions and reproduction

```json
{
  "mlx": "0.32.2",
  "mlx-metal": "0.32.2",
  "numpy": "2.5.3",
  "torch": "2.14.0",
  "transformers": "5.17.0",
  "tokenizers": "0.23.2",
  "huggingface-hub": "1.32.0"
}
```

```bash
uv sync --extra dev --extra reference --extra benchmark --extra demo
source .venv/bin/activate
gh repo clone NandhaKishorM/laya .upstream
git -C .upstream checkout 573e5b62696ba441230cd6be71d593331b5d23af
python -m benchmarks.download
pytest -q
python -m benchmarks.validate --repeats 100
python -m benchmarks.run --iterations 50 --warmup 5
python -m benchmarks.accuracy --per-class 64
python -m benchmarks.report
```

The benchmark uses original upstream safetensors and explicitly pinned Hugging Face revisions from `benchmarks/common.py`. Run GPU commands sequentially. The checked-in `uv.lock` captures the dependency environment. `benchmarks/results/` contains the raw measurements; no downloaded model weights are committed.
