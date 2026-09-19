"""Build BENCHMARKS.md and a standalone chart from the recorded JSON samples."""

import argparse
import json
from pathlib import Path

from .common import MODELS, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()
    records = {}
    variants = [("torch-mps", "float32"), ("mlx", "float32"), ("mlx", "float16")]
    for name in MODELS:
        for backend, dtype in variants:
            file = args.results / f"{name}-{backend}-{dtype}.json"
            records[name, backend, dtype] = json.loads(file.read_text())
    first = next(iter(records.values()))
    env = first["environment"]

    def row(name, variant, count, workload):
        report = records[name, *variant]
        return next(
            x for x in report["results"] if x["questions"] == count and x["workload"] == workload
        )

    # Every backend must have processed byte-identical inputs and identical token counts.
    for name in MODELS:
        for work, count in [("short", n) for n in (1, 5, 10, 50)] + [("long", n) for n in (1, 10)]:
            selected = [row(name, v, count, work) for v in variants]
            assert len({x["input_sha256"] for x in selected}) == 1
            assert len({(x["input_tokens"], x["sequence_length"]) for x in selected}) == 1
    lines = [
        "# Local benchmarks",
        "",
        f"Measured on **{env['cpu']}**, 40-core GPU, **{env['unified_memory_bytes'] / 1024**3:.0f} GiB unified memory**, {env['os']}, Python {env['python']}.",
        "",
        "These are local measurements of the native MLX port and the pinned upstream PyTorch runtime on the same Mac. They are not comparisons with the upstream T4 or third-party API figures.",
        "",
        "## Method",
        "",
        f"- Each checkpoint/backend runs in a fresh process, sequentially, with {first['warmup']} warmup iterations and {first['iterations']} timed iterations per workload and timing mode.",
        "- End-to-end timing includes prompt construction, tokenization, tensor construction, model execution, calibration, and result formatting. Model loading and downloads are excluded.",
        "- Forward timing uses prepared device tensors and includes the encoder, decision head, scoring head and action head. GPU completion is synchronized for every sample; lazy MLX graph construction alone is never timed as inference.",
        "- All backends receive identical state and question JSON. The report generator verifies matching input hashes, token totals and padded sequence lengths.",
        "- MLX uses batch_size=64 for these measurements so even 50 questions fit in one batch. The public runtime defaults to 16 to bound memory; changing batch size can change throughput.",
        "- PyTorch MPS uses upstream's default FP32. MLX FP32 provides the same-precision comparison. MLX FP16 trades some numerical precision for speed and memory; its speedup includes that precision change.",
        "- P50/P95 are percentiles of measured wall-clock latency. Throughput is questions / mean latency, not the inverse of P50. Raw JSON contains every timing sample.",
        "- This is one development machine and one run of each configuration, with normal OS activity. No claim of cross-device performance or production endurance is made.",
        "",
        "![Local latency comparison](benchmarks/latency.png)",
        "",
        "## End-to-end short input latency",
        "",
        "All values are milliseconds per request. P95 is shown after `/`.",
        "",
        "| Checkpoint | Questions | Tokens / padded length | PyTorch MPS FP32 P50 / P95 | MLX FP32 P50 / P95 | MLX FP16 P50 / P95 | FP16 questions/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MODELS:
        for count in (1, 5, 10, 50):
            selected = [row(name, v, count, "short") for v in variants]
            metrics = [x["end_to_end"] for x in selected]
            text = " | ".join(f"{m['p50_ms']:.2f} / {m['p95_ms']:.2f}" for m in metrics)
            lines.append(
                f"| {name} | {count} | {selected[0]['input_tokens']} / {selected[0]['sequence_length']} | {text} | {metrics[-1]['questions_per_second']:.1f} |"
            )
    lines += [
        "",
        "## Full-context latency",
        "",
        "Long input fills each checkpoint's configured limit, including question and option tokens.",
        "",
        "| Checkpoint | Questions | Padded length | PyTorch MPS FP32 P50 | MLX FP32 P50 | MLX FP16 P50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in MODELS:
        for count in (1, 10):
            selected = [row(name, v, count, "long") for v in variants]
            text = " | ".join(f"{x['end_to_end']['p50_ms']:.2f}" for x in selected)
            lines.append(f"| {name} | {count} | {selected[0]['sequence_length']} | {text} |")
    lines += [
        "",
        "## Memory",
        "",
        "MLX peak allocated memory includes model weights, inputs and intermediates; cache memory is recorded separately. MPS current tensor/driver allocations in the raw JSON are different metrics and are not presented as comparable peaks.",
        "",
        "| Checkpoint | Parameters | FP16 weights (MiB) | FP16 peak: 1 short question (MiB) | FP16 peak: 10 full-context questions (MiB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in MODELS:
        record = records[name, "mlx", "float16"]
        short = row(name, ("mlx", "float16"), 1, "short")
        long = row(name, ("mlx", "float16"), 10, "long")
        lines.append(
            f"| {name} | {record['parameter_count']:,} | {record['weight_bytes'] / 1024**2:.1f} | {short['memory']['mlx_peak_bytes'] / 1024**2:.1f} | {long['memory']['mlx_peak_bytes'] / 1024**2:.1f} |"
        )
    validation = json.loads((args.results / "validation.json").read_text())
    lines += [
        "",
        "## Numerical parity and stability",
        "",
        "Real checkpoint validation covers 16 cases and 63 questions per checkpoint and precision: eight languages, empty and long states, conversation lists, mask-token literals, structured criteria, mixed question batches and 20 options. This measures fidelity to upstream, not correctness of every model answer.",
        "",
        "| Checkpoint | Precision | Argmax agreement | Max calibrated probability error | Repeated calls | Active memory growth (bytes) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in validation["results"]:
        lines.append(
            f"| {result['model']} | {result['dtype']} | {result['argmax_agreements']}/{result['questions']} | {result['probability_max_abs_error']:.7f} | {result['stability']['calls']} | {result['stability']['active_growth_bytes']} |"
        )
    lines += [
        "",
        "Every repeated call checks finite logits and exactly repeatable public JSON for the same input and batch shape. Memory growth is measured after garbage collection and clearing the MLX cache. The per-case JSON also records raw action-logit error: action logits can have large magnitudes, so action softmax probabilities are checked separately. FP32 probability tolerance is 0.0001 and FP16 tolerance is 0.02; these thresholds were set in the validation script before measuring.",
        "",
    ]
    accuracy_file = args.results / "accuracy.json"
    if accuracy_file.exists():
        accuracy = json.loads(accuracy_file.read_text())
        lines += [
            "## Labeled task sample",
            "",
            f"AG News test split: {accuracy['count']} examples, equal class counts, seed {accuracy['seed']}. This is a small English classification sample; AG News appears in upstream's training mix, and the typed-decisions checkpoint targets different tasks. Source revision, sampled indices, gold labels and both backends' predictions are in `benchmarks/results/accuracy.json`. No sample text is redistributed.",
            "",
            "| Checkpoint | Upstream MPS FP32 accuracy | MLX FP16 accuracy | Prediction agreement |",
            "|---|---:|---:|---:|",
        ]
        for result in accuracy["results"]:
            lines.append(
                f"| {result['model']} | {result['upstream_accuracy']:.4f} | {result['mlx_accuracy']:.4f} | {result['port_agreements']}/{accuracy['count']} |"
            )
        lines.append("")
    lines += [
        "## Versions and reproduction",
        "",
        "```json",
        json.dumps(env["packages"], indent=2),
        "```",
        "",
        "```bash",
        "uv sync --extra dev --extra reference --extra benchmark",
        "source .venv/bin/activate",
        "gh repo clone NandhaKishorM/laya .upstream",
        f"git -C .upstream checkout {env['upstream_revision']}",
        "python -m benchmarks.download",
        "pytest -q",
        "python -m benchmarks.validate --repeats 100",
        "python -m benchmarks.run --iterations 50 --warmup 5",
        "python -m benchmarks.accuracy --per-class 64",
        "python -m benchmarks.report",
        "```",
        "",
        "The benchmark uses original upstream safetensors and explicitly pinned Hugging Face revisions from `benchmarks/common.py`. Run GPU commands sequentially. The checked-in `uv.lock` captures the dependency environment. `benchmarks/results/` contains the raw measurements; no downloaded model weights are committed.",
        "",
    ]
    (ROOT / "BENCHMARKS.md").write_text("\n".join(lines))
    plot(records, row)


def plot(records, row):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    variants = [("torch-mps", "float32"), ("mlx", "float32"), ("mlx", "float16")]
    labels = ["PyTorch MPS FP32", "MLX FP32", "MLX FP16"]
    colors = ["#8894A5", "#4378B9", "#13866F"]
    x = np.arange(3)
    for ax, count in zip(axes, (1, 10)):
        for i, (variant, label, color) in enumerate(zip(variants, labels, colors)):
            values = [row(n, variant, count, "short")["end_to_end"] for n in MODELS]
            median = np.array([v["p50_ms"] for v in values])
            upper = np.array([v["p95_ms"] for v in values]) - median
            ax.bar(
                x + (i - 1) * 0.25,
                median,
                0.23,
                label=label,
                color=color,
                yerr=np.stack([np.zeros(3), upper]),
                capsize=2,
            )
        ax.set_xticks(x, ["English", "Multilingual", "Typed decisions"])
        ax.set_title(f"{count} question{'s' if count > 1 else ''} per request", weight="bold")
        ax.set_ylabel("End-to-end latency (ms), lower is faster")
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Laya on Apple M3 Max · 40-core GPU · 128 GB", fontsize=15, weight="bold")
    fig.supxlabel(
        "P50 bars with P95 whiskers · 5 warmups + 50 timed calls · short input · batch size limit 64",
        fontsize=9,
    )
    fig.savefig(ROOT / "benchmarks/latency.png", dpi=180)
    fig.savefig(ROOT / "benchmarks/latency.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
