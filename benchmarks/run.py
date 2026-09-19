"""Run benchmarks sequentially in isolated processes, with reproducible inputs."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .common import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "USE_TF": "0"}
    for name in args.models:
        for backend, dtype in (("torch-mps", "float32"), ("mlx", "float32"), ("mlx", "float16")):
            output = args.output / f"{name}-{backend}-{dtype}.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.worker",
                    "--model",
                    name,
                    "--model-root",
                    str(args.model_root),
                    "--backend",
                    backend,
                    "--dtype",
                    dtype,
                    "--iterations",
                    str(args.iterations),
                    "--warmup",
                    str(args.warmup),
                    "--output",
                    str(output),
                ],
                env=env,
                check=True,
            )


if __name__ == "__main__":
    main()
