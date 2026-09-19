"""Prepare validated, documented MLX exports for a subsequent `hf upload`.

This command only writes local artifacts. It does not create or upload Hub repos.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from benchmarks.common import MODELS, ROOT, UPSTREAM_REVISION, save_json
from laya_mlx.convert import convert
from laya_mlx.model import sanitize_weights


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_tensors(source, destination):
    """Verify every exported tensor against the original, not a random sample."""
    with (
        safe_open(source, framework="numpy") as original,
        safe_open(destination, framework="numpy") as exported,
    ):
        mapping = sanitize_weights({key: key for key in original.keys()})
        if set(mapping) != set(exported.keys()):
            raise ValueError("Exported parameter names do not match the original checkpoint")
        for converted_name, original_name in mapping.items():
            np.testing.assert_array_equal(
                exported.get_tensor(converted_name),
                original.get_tensor(original_name).astype(np.float16),
            )
        return len(mapping)


def model_card(name, repo, results):
    multilingual = name == "laya-multilingual"
    context = 512 if name == "laya" else 1024
    encoder = "mmBERT-base" if multilingual else "ModernBERT-large"
    fp16 = next(r for r in results if r["dtype"] == "float16")
    languages = "tags:\n- multilingual\n" if multilingual else "language:\n- en\ntags:\n"
    return f"""---
license: apache-2.0
library_name: mlx
pipeline_tag: text-classification
base_model: convaiinnovations/{name}
{languages}- mlx
- laya
- modernbert
- apple-silicon
- decision-model
---

# {repo.split("/")[-1]}

Native **MLX FP16** conversion of [convaiinnovations/{name}](https://huggingface.co/convaiinnovations/{name}) for Apple silicon.

This checkpoint uses **{encoder}**, a **{context}-token total context**, and Laya's decision Transformer, scoring head and action head. It supports `choice`, ordinal `score`, and boolean `noul` questions. All model computation runs in MLX; the runtime does not require PyTorch or Transformers.

## Usage

Install the dedicated runtime on an Apple silicon Mac with macOS 26+ and Python 3.11+:

```bash
python -m pip install 'git+https://github.com/mizorewww/laya-mlx.git'
```

```python
import laya_mlx as laya

agent = laya.load("{repo}")
result = agent.predict(
    "I was billed twice. Please refund the duplicate today.",
    {{
        "department": {{
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": ["billing", "technical", "sales"],
        }},
        "refund": {{
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
        }},
    }},
)
print(result["answers"])
```

Use `dtype="float32"` for closer agreement with upstream FP32 arithmetic. The source weights themselves are FP16. Question formatting, tokenizer behavior, calibration temperatures and output schema are preserved.

This is a bidirectional decision encoder loaded with `laya_mlx`. The package provides the custom architecture needed to interpret the checkpoint. The repository does not include a generative language model or training implementation.

## Validation

Tested locally on Apple M3 Max, 40-core GPU, 128 GB unified memory, macOS 27.2, Python 3.12.13 and MLX 0.32.2.

- FP16 agrees with upstream PyTorch MPS FP32 on the argmax of **{fp16["argmax_agreements"]}/{fp16["questions"]}** decision distributions across 16 cases.
- Maximum calibrated probability difference: **{fp16["probability_max_abs_error"]:.7f}**.
- **{fp16["stability"]["calls"]} repeated calls** produced finite, deterministic public outputs; measured MLX active-memory growth after clearing caches was **{fp16["stability"]["active_growth_bytes"]} bytes**.
- Every exported tensor was checked for exact equality with the corresponding source tensor cast to FP16.

The included `validation.json` contains numerical and stability measurements for both FP32 and FP16 arithmetic. [Full performance report and raw timing samples](https://github.com/mizorewww/laya-mlx/blob/main/BENCHMARKS.md) compare MLX with the original runtime on the same machine. These checks establish port fidelity, not that every model answer is correct.

## Provenance and limits

- Source checkpoint: `convaiinnovations/{name}` at `{MODELS[name]}`.
- Upstream code: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), commit `{UPSTREAM_REVISION}`.
- Conversion changes parameter names for MLX and preserves FP16 weights. It does not retrain or quantize to fewer bits.
- This is an independent port. Model quality, calibration and language/task limitations remain those of the original checkpoint. Questions and options share the context budget with the input state.
- The typed-decisions checkpoint is specialized for upstream workflows; the multilingual checkpoint is the intended choice for non-English text.

Apache-2.0. Original Laya models and code are by Convai Innovations and contributors. See `LICENSE`, `NOTICE`, `mlx_config.json` and `manifest.json` for attribution and export details.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", required=True)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("models/hub"))
    args = parser.parse_args()
    validation = json.loads((ROOT / "benchmarks/results/validation.json").read_text())
    # Conversion is tensor loading/casting; it can use CPU without competing for GPU execution.
    mx.set_default_device(mx.cpu)
    for name, revision in MODELS.items():
        repo = f"{args.account}/{name}-mlx"
        destination = args.output / f"{name}-mlx"
        convert(args.model_root / name, destination, dtype="float16", revision=revision)
        metadata = json.loads((destination / "mlx_config.json").read_text())
        metadata.update(
            source=f"convaiinnovations/{name}", source_revision=revision, repository=repo
        )
        save_json(destination / "mlx_config.json", metadata)
        verified = verify_tensors(
            args.model_root / name / "model.safetensors", destination / "model.safetensors"
        )
        results = [r for r in validation["results"] if r["model"] == name]
        (destination / "README.md").write_text(model_card(name, repo, results))
        for file in ("LICENSE", "NOTICE"):
            shutil.copyfile(ROOT / file, destination / file)
        save_json(destination / "validation.json", {"results": results})
        manifest = {
            "repository": repo,
            "source": f"convaiinnovations/{name}",
            "source_revision": revision,
            "format": "laya-mlx",
            "dtype": "float16",
            "verified_tensors": verified,
            "files": {
                str(p.relative_to(destination)): {
                    "bytes": p.stat().st_size,
                    "sha256": file_sha256(p),
                }
                for p in sorted(destination.rglob("*"))
                if p.is_file()
            },
        }
        save_json(destination / "manifest.json", manifest)
        print(f"Prepared {repo}: {verified} tensors verified; {destination}", flush=True)


if __name__ == "__main__":
    main()
