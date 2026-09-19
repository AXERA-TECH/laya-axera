"""Small CPU-only spectrum probe of four selected matrices, not model inference.

Single-thread BLAS limits are established before importing NumPy. The largest
eigensystem is 1024 x 1024. No full model is loaded. Sampled matrices are named
explicitly, so these findings cannot be mistaken for a whole-model survey.
"""

import hashlib
import json
import os
import time
from pathlib import Path

for setting in (
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
):
    os.environ[setting] = "1"

# NumPy/BLAS must be imported only after the process thread limits are set.
import numpy as np  # noqa: E402
from safetensors import safe_open  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = [
    ("laya", "encoder.layers.14.attn.Wo.weight"),
    ("laya", "encoder.layers.14.mlp.Wi.weight"),
    ("laya-multilingual", "encoder.layers.11.attn.Wo.weight"),
    ("laya-multilingual", "encoder.layers.11.mlp.Wi.weight"),
]


def probe(model, key):
    begin = time.perf_counter()
    with safe_open(ROOT / "models" / model / "model.safetensors", framework="numpy") as checkpoint:
        original = checkpoint.get_tensor(key)
    original_hash = hashlib.sha256(original.tobytes()).hexdigest()
    weight = original.astype(np.float64)
    m, n = weight.shape
    gram = weight.T @ weight if n <= m else weight @ weight.T
    energies = np.maximum(np.linalg.eigvalsh(gram)[::-1], 0)
    cumulative = np.minimum(np.cumsum(energies) / energies.sum(), 1.0)
    # W: m x n -> U:m x r, V:r x n. Cost ratio r*(m+n)/(m*n).
    tenfold_rank = max(1, int(m * n / (10 * (m + n))))
    costs = []
    for rank in sorted({tenfold_rank, 32, 64, 128, 256, min(m, n)}):
        if rank > min(m, n):
            continue
        kept = float(cumulative[rank - 1])
        costs.append(
            {
                "rank": rank,
                "matrix_flop_ratio": rank * (m + n) / (m * n),
                "retained_squared_frobenius_fraction": kept,
                "best_relative_frobenius_error": float(np.sqrt(max(0, 1 - kept))),
            }
        )
    return {
        "model": model,
        "weight": key,
        "shape": [m, n],
        "checkpoint_dtype": str(original.dtype),
        "matrix_sha256": original_hash,
        "method": "float64 Gram symmetric eigensolver on CPU; no randomized approximation",
        "largest_singular_value": float(np.sqrt(energies[0])),
        "smallest_singular_value": float(np.sqrt(energies[-1])),
        "stable_rank": float(energies.sum() / energies[0]),
        "rank_for_90_percent_squared_frobenius": int(np.searchsorted(cumulative, 0.9) + 1),
        "rank_for_99_percent_squared_frobenius": int(np.searchsorted(cumulative, 0.99) + 1),
        "rank_for_99_9_percent_squared_frobenius": int(np.searchsorted(cumulative, 0.999) + 1),
        "max_rank_for_10x_matrix_flop_reduction": tenfold_rank,
        "ranks": costs,
        "cpu_seconds": time.perf_counter() - begin,
    }


def main():
    records = []
    for model, key in SAMPLES:
        record = probe(model, key)
        records.append(record)
        print(json.dumps(record, separators=(",", ":")), flush=True)
    output = {
        "kind": "CPU spectrum measurement of four selected matrices; not task accuracy or GPU latency",
        "numpy_version": np.__version__,
        "thread_limits": {
            setting: os.environ[setting]
            for setting in (
                "VECLIB_MAXIMUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
            )
        },
        "samples": records,
    }
    (ROOT / "experiments/math_spectrum.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
