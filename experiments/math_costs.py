"""CPU-only static work budgets for the unchanged Laya architecture.

No MLX or Torch import, no inference and no performance measurement. Run from
the repository root with `.venv/bin/python experiments/math_costs.py`.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("laya", "laya-multilingual", "laya-typed-decisions")
BANDWIDTH = 400_000_000_000  # Advertised 40-GPU-core M3 Max unified bandwidth.


def model_budget(name):
    folder = ROOT / "models" / name
    cfg = json.loads((folder / "encoder/config.json").read_text())
    agent = json.loads((folder / "rl_agent_config.json").read_text())
    d, i, n, h = (
        cfg["hidden_size"],
        cfg["intermediate_size"],
        cfg["num_hidden_layers"],
        agent["head_layers"],
    )
    types = cfg["layer_types"]
    ng = types.count("full_attention")
    nl = types.count("sliding_attention")
    a = n * (4 * d * d + 3 * d * i) + h * 12 * d * d
    radius = cfg["local_attention"] // 2
    benchmark_path = ROOT / "benchmarks/results" / f"{name}-mlx-float16.json"
    benchmark_bytes = benchmark_path.read_bytes()
    benchmark = json.loads(benchmark_bytes)
    rows = []
    for result in benchmark["results"]:
        b, length = result["questions"], result["sequence_length"]
        dense = 2 * b * length * a
        attention = 4 * b * (n + h) * length * length * d
        # Exact count, including shortened windows at the sequence boundaries.
        local_pairs = sum(
            min(length - 1, p + radius) - max(0, p - radius) + 1 for p in range(length)
        )
        local_saving = 4 * b * nl * (length * length - local_pairs) * d
        # Illustrative fixed R=5 includes CLS plus four marker positions.
        r = min(5, length)
        head_saving = 20 * b * (length - r) * d * d + 4 * b * length * (length - r) * d
        total = dense + attention
        latency = result["end_to_end"]["p50_ms"]
        target = latency / 10
        rows.append(
            {
                "workload": result["workload"],
                "questions": b,
                "sequence_length": length,
                "end_to_end_p50_ms": latency,
                "tenfold_target_ms": target,
                "dense_gflops": dense / 1e9,
                "dense_attention_gflops": attention / 1e9,
                "total_gflops": total / 1e9,
                "baseline_effective_tflops": total / (latency * 1e9),
                "tenfold_required_tflops": total / (target * 1e9),
                "attention_flop_fraction": attention / total,
                "exact_local_flop_reduction": local_saving / total,
                "exact_final_head_r5_flop_reduction": head_saving / total,
                "combined_exact_flop_reduction": (local_saving + head_saving) / total,
                "padded_token_fraction": 1 - result["input_tokens"] / (b * length),
                "hypothetical_fp16_onchip_resident_bytes_needed_for_target": max(
                    0, 2 * a - BANDWIDTH * target / 1000
                ),
            }
        )
    return {
        "hidden_size": d,
        "intermediate_size": i,
        "encoder_layers": n,
        "head_layers": h,
        "global_encoder_layers": ng,
        "local_encoder_layers": nl,
        "main_dense_weight_count": a,
        "embedding_weight_count": cfg["vocab_size"] * d,
        "encoder_mlp_fraction_of_main_dense": n * 3 * d * i / a,
        "first_layer_qkv_fraction_of_main_dense": 3 * d * d / a,
        "fp16_main_dense_weight_bytes": 2 * a,
        "fp16_weight_stream_lower_ms": 2 * a / BANDWIDTH * 1000,
        "affine4_g64_weight_stream_lower_ms": a * (0.5 + 4 / 64) / BANDWIDTH * 1000,
        "affine8_g64_weight_stream_lower_ms": a * (1 + 4 / 64) / BANDWIDTH * 1000,
        "benchmark_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
        "source_sha256": benchmark["environment"]["source_sha256"],
        "checkpoint_revision": benchmark["revision"],
        "rows": rows,
    }


def student_budget(name, n_student, d_student, i_student, h_student=1):
    source = model_budget(name)
    ds, ins = d_student, i_student
    a_student = n_student * (4 * ds * ds + 3 * ds * ins) + h_student * 12 * ds * ds
    return {
        "teacher": name,
        "student_encoder_layers": n_student,
        "student_hidden_size": ds,
        "student_intermediate_size": ins,
        "student_head_layers": h_student,
        "student_main_dense_weights": a_student,
        "teacher_to_student_dense_flop_ratio_same_tokens": source["main_dense_weight_count"]
        / a_student,
    }


def main():
    data = {
        "kind": "static arithmetic and conditional bandwidth estimates, not new timings",
        "flop_convention": "multiply and add count separately; excludes norms, activations, scoring, masks and traffic",
        "bandwidth_bytes_per_second": BANDWIDTH,
        "bandwidth_source": "https://support.apple.com/en-us/117737",
        "models": {name: model_budget(name) for name in MODELS},
        "student_hypotheses": [
            student_budget("laya", 6, 512, 1344),
            student_budget("laya", 4, 512, 1344),
            student_budget("laya-multilingual", 4, 384, 576),
            student_budget("laya-multilingual", 6, 384, 576),
        ],
        "amdahl_tenfold_hotspot_requirements": [
            {"wall_time_fraction": f, "hotspot_speedup_required": f / (f - 0.9)}
            for f in (0.95, 0.98, 0.99)
        ],
    }
    out = ROOT / "experiments/math_costs.json"
    out.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    for name, data in data["models"].items():
        print(
            name,
            "A",
            data["main_dense_weight_count"],
            "stream lower ms",
            data["fp16_weight_stream_lower_ms"],
        )
        for row in data["rows"]:
            print(
                row["workload"],
                row["questions"],
                row["sequence_length"],
                "p50",
                round(row["end_to_end_p50_ms"], 3),
                "target",
                round(row["tenfold_target_ms"], 3),
                "GF",
                round(row["total_gflops"], 2),
                "required TF/s",
                round(row["tenfold_required_tflops"], 2),
                "exact reduction %",
                round(100 * row["combined_exact_flop_reduction"], 3),
            )


if __name__ == "__main__":
    main()
