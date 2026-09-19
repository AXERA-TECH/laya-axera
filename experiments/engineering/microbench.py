"""Small serial GPU probes for GEMM and exact-erf GELU/gate fusion."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from benchmarks.common import environment, save_json

from .kernels import metal_gelu_gate
from .run_variants import measured


def original(x):
    value, gate = mx.split(x, 2, axis=-1)
    return nn.gelu(value) * gate


def main():
    mx.random.seed(20260919)
    report = {"environment": environment(), "gemm": [], "gelu_gate": []}
    for m, n, k in [
        (78, 5248, 1024),
        (1312, 5248, 1024),
        (512, 5248, 1024),
        (4096, 5248, 1024),
        (79, 2304, 768),
        (1312, 2304, 768),
        (1024, 2304, 768),
        (8192, 2304, 768),
    ]:
        x = mx.random.normal((m, k)).astype(mx.float16)
        w = mx.random.normal((n, k)).astype(mx.float16)
        mx.eval(x, w)
        for _ in range(5):
            mx.eval(x @ w.T)
        timing = measured(lambda: x @ w.T, 30)
        row = {
            "m_n_k": [m, n, k],
            "dtype": "float16",
            "timing": timing,
            "achieved_tflops": 2 * m * n * k / (timing["p50_ms"] * 1e9),
        }
        report["gemm"].append(row)
        print("gemm", [m, n, k], row["achieved_tflops"], flush=True)
    functions = {
        "eager_gelu_then_gate": original,
        "compiled_gelu_gate": mx.compile(original),
        "metal_gelu_gate": metal_gelu_gate,
    }
    for tokens, intermediate in [
        (78, 2624),
        (1312, 2624),
        (512, 2624),
        (4096, 2624),
        (79, 1152),
        (1312, 1152),
        (1024, 1152),
        (8192, 1152),
    ]:
        x = mx.random.uniform(low=-8, high=8, shape=(tokens, 2 * intermediate)).astype(mx.float16)
        mx.eval(x)
        reference = original(x)
        mx.eval(reference)
        reference = np.asarray(reference).copy()
        row = {"tokens_intermediate": [tokens, intermediate], "candidates": {}}
        for name, function in functions.items():
            cold = measured(lambda: function(x), 1)
            for _ in range(5):
                mx.eval(function(x))
            timing = measured(lambda: function(x), 40)
            actual = function(x)
            mx.eval(actual)
            actual = np.asarray(actual)
            row["candidates"][name] = {
                "cold_call": cold,
                "timing": timing,
                "max_absolute_error": float(
                    np.max(np.abs(actual.astype(np.float32) - reference.astype(np.float32)))
                ),
                "elements_different": int(np.sum(actual != reference)),
                "elements": actual.size,
            }
        report["gelu_gate"].append(row)
        print(
            "gelu",
            [tokens, intermediate],
            {
                k: (v["timing"]["p50_ms"], v["max_absolute_error"])
                for k, v in row["candidates"].items()
            },
            flush=True,
        )
    save_json(Path("experiments/engineering/microbench.json"), report)
    print(json.dumps({"gemm_probes": len(report["gemm"]), "gelu_probes": len(report["gelu_gate"])}))


if __name__ == "__main__":
    main()
