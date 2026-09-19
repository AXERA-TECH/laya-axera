# Snake speed and stability: M3 Max

The final **truecolor** campaign completed **8,160 moves with zero deaths and 4 safety interventions**. The default model was `aac6fef/laya-multilingual-mlx` in FP16, with compact planner descriptions, a 24 × 16 board and initial length 6. These measurements use the default eager runtime; opt-in compilation is evaluated separately in [Snake optimization](SNAKE_OPTIMIZATION.md).

**Sustained uncapped throughput: 63.61 moves/second overall across 2,400 steps.** Each of four seeds completed 600 moves. Their rates ranged from 46.32 to 76.37 moves/second. Every game survived, maintained body cycle order and kept reaching food. The safety layer corrected 2 raw model proposals in these episodes.

**Highest passing tested paced computation budget: 20 FPS.** All four seeds met the requirement that at least 99% of active ticks fit within 50 ms. The actual wall-clock rate, including OS sleep overshoot, was 18.69–18.94 moves/second. This is a budget setting, not a claim of exactly 20 rendered frames every second.

## Sustained maximum speed

| Seed | Steps | Actual steps/s | Active tick p50 / p99 (ms) | Score / length | Interventions |
| --- | --- | --- | --- | --- | --- |
| 101 | 600 | 46.32 | 15.52 / 39.67 | 20 / 26 | 1 |
| 102 | 600 | 67.91 | 13.84 / 22.96 | 24 / 30 | 0 |
| 103 | 600 | 74.20 | 12.27 / 21.42 | 23 / 29 | 1 |
| 104 | 600 | 76.37 | 11.98 / 21.00 | 16 / 22 | 0 |

Combined uncapped model-inference p50 / p95 / p99 was **10.08 / 31.68 / 33.61 ms**. Complete active-tick p50 / p99 was **12.70 / 37.21 ms**. An active tick includes planning, synchronized inference, Rich composition, truecolor ANSI serialization and the game update. No pacing delay is inserted.

## Fixed-budget stability

| Seed | Target FPS | Steps | Actual steps/s | Active tick p99 (ms) | Budget misses |
| --- | --- | --- | --- | --- | --- |
| 101 | 20.0 | 600 | 18.69 | 39.93 | 0 (0.00%) |
| 102 | 20.0 | 600 | 18.86 | 45.09 | 0 (0.00%) |
| 103 | 20.0 | 600 | 18.94 | 48.66 | 6 (1.00%) |
| 104 | 20.0 | 600 | 18.85 | 44.53 | 0 (0.00%) |

The passing test had **6 late active ticks out of 2,400 (99.75% within budget overall)**. The worst seed had 6/600 late ticks, exactly the predefined 1% limit. All moves wait for a fresh model result, so slow inference reduces the game rate instead of executing stale actions.

## Short sweep

Each candidate below ran 120 moves with seed 7. A short passing probe selects a candidate for the longer four-seed test; it is not sufficient by itself to claim stability. All games survived, including timing failures.

| Target FPS | Actual steps/s | Active tick p99 (ms) | Budget misses | Short sweep |
| --- | --- | --- | --- | --- |
| 10.0 | 9.60 | 52.13 | 0.00% | pass |
| 12.0 | 11.39 | 42.07 | 0.00% | pass |
| 15.0 | 14.10 | 48.69 | 0.00% | pass |
| 18.0 | 16.95 | 58.10 | 2.50% | fail |
| 20.0 | 18.84 | 43.84 | 0.00% | pass |
| 25.0 | 24.23 | 45.66 | 5.83% | fail |
| 30.0 | 28.60 | 40.04 | 97.50% | fail |
| 35.0 | 28.69 | 40.12 | 100.00% | fail |
| 40.0 | 28.16 | 44.15 | 100.00% | fail |
| 45.0 | 26.70 | 45.56 | 100.00% | fail |
| 50.0 | 28.67 | 40.23 | 100.00% | fail |
| 60.0 | 28.89 | 40.97 | 100.00% | fail |

The short sweep is nonmonotonic, and measured latency changes considerably over time. Paced and uncapped runs differ in device idle periods; normal desktop activity, scheduling and clock behavior were not isolated experimentally. The data do not identify a single cause or a universal speed ceiling. We report the achieved rate, all samples and the highest passing **tested** setting.

## What the stability result includes

Laya receives planner features, including legal directions and the best admissible progress toward food. It computes real probabilities. The cycle safety layer may correct execution while retaining the raw probability bars and counting every intervention. The checkpoint was not trained on Snake here. [Exact UI metric meanings and policy behavior](SNAKE_DEMO.md#what-the-ai-does).

A separate raw top-1 control ran seeds 101–103 for 600 moves each with the execution shield disabled. All three survived; scores were **9, 24 and 19**, compared with **20, 24 and 23** in the shielded counterparts. The raw control still supplies planner features. Its survival over this horizon does not establish indefinite unshielded survival or unaided board reasoning.

The published showcase is a separate **100.005-second actual TTY run**, at a 12 FPS target: **1,144 moves, score 40, length 46, zero deaths and zero interventions**. Achieved throughput was 11.44 moves/second. Every recorded board and action is checked by deterministic replay in the test suite. The visible inference numbers come from that run, rather than being replaced with favorable benchmark numbers.

A separate optimized maximum-speed TTY recording completed **1,296 moves in 20.01 seconds (64.77 moves/second)**, with score 44, length 50, zero deaths and zero interventions. This includes the actual terminal stream writes. Its [15-second original-speed video](assets/snake-fast.mp4) and [complete recording](../benchmarks/results/snake-fast.jsonl) are included alongside the slower presentation clip.

## Measurement boundaries and provenance

- Apple M3 Max, 40 GPU cores, 128 GiB unified memory; macOS 27.2, Python 3.12.13.
- MLX 0.32.2, NumPy 2.5.3, Rich 15.0.0, tokenizers 0.23.2, huggingface-hub 1.32.0.
- Original FP16 weights, with no quantization or custom kernel. Upstream model revision: `052592a15d198d9ad47da779604259b10b47b7aa`.
- Weight SHA-256 from the previously verified published manifest: `7fc5834af4d8fdfb268d272a9d1a66e5819a0daac98241651c4c888cc43adff1`.
- Each move calls `Agent.predict` once with three batched questions. Inference timing includes tokenization, synchronized MLX evaluation and output conversion.
- The benchmark explicitly enables true color, independent of `NO_COLOR`, and serializes Rich output into an in-memory stream. The terminal emulator's screen painting, model loading and warmup are excluded. Actual terminal controls and cleanup were tested separately.
- The report stores configuration, environment, source fingerprint and every per-step probability, execution decision and timing. The later addition of an output-token label and opt-in optimization does not change the baseline model or game policy.

## Earlier runs are retained

The first detailed-prompt and compact-prompt campaigns inherited `NO_COLOR=1` from the tool environment. They measured monochrome serialization and are retained as exploratory results, rather than substituted for the final color benchmark. The earlier compact campaign completed 12,360 moves without a death and measured 52.55 uncapped moves/second overall; its highest passing tested budget was 18 FPS. The difference from the final run is **not** attributed to enabling color: the machine's timings vary substantially.

The detailed-prompt 20 FPS soak failed its timing criterion on all four seeds, while every game survived. The earlier compact campaign's higher-rate failures are also retained. No failed completed episode has been removed from those files.

## Files and reproduction

- [Final truecolor traces](../benchmarks/results/snake.json)
- [Earlier compact monochrome campaign](../benchmarks/results/snake-compact-no-color.json)
- [Earlier detailed-prompt monochrome campaign](../benchmarks/results/snake-detailed-prompt.json)
- [Actual TTY controls and offline inference check](../benchmarks/results/snake-integration.json)
- [Original showcase recording](../benchmarks/results/snake-showcase.jsonl)
- [Video provenance](assets/snake-demo.json)
- [Compilation, prefix and prompt experiments](SNAKE_OPTIMIZATION.md)

```bash
uv run --extra demo laya-snake benchmark \
  --rates 10,12,15,18,20,25,30,35,40,45,50,60 \
  --raw-steps 600 --sweep-steps 120 --soak-steps 600 \
  --seeds 101,102,103,104 --output artifacts/snake/benchmark.json
```

Use `--resume` with the same output path after an interruption. Live presentation defaults to a readable 12 FPS target; `--max-speed` measures the current continuous decision rate.
