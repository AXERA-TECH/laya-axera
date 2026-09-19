"""Small labeled AG News test: stock upstream MPS FP32 versus native MLX FP16.

This measures port agreement and a reproducible task accuracy sample, not the
upstream project's complete evaluation or a new claim of general model quality.
"""

import argparse
import gc
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from laya_mlx import Agent

from .common import MODELS, digest, environment, load_reference, save_json

DATASET = "fancyzhx/ag_news"
REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
QUESTIONS = {
    "topic": {
        "type": "choice",
        "instructions": "Classify the topic of this news article.",
        "criteria": {
            "World": "world news, international affairs and politics",
            "Sports": "sports, athletes and competitions",
            "Business": "business, companies, finance and economics",
            "Sci/Tech": "science, technology, computers and research",
        },
    }
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--per-class", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/accuracy.json"))
    args = parser.parse_args()
    if not 1 <= args.per_class <= 1900:
        parser.error("--per-class must be between 1 and 1900")
    file = hf_hub_download(
        DATASET, filename="data/test-00000-of-00001.parquet", repo_type="dataset", revision=REVISION
    )
    data = pq.read_table(file).to_pydict()
    labels = np.array(data["label"])
    rng = np.random.default_rng(args.seed)
    indices = np.concatenate(
        [rng.choice(np.flatnonzero(labels == i), args.per_class, replace=False) for i in range(4)]
    )
    rng.shuffle(indices)
    texts = [data["text"][i] for i in indices]
    gold = [int(labels[i]) for i in indices]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment(),
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "split": "test",
        "seed": args.seed,
        "sampling": "without replacement, equal samples per class",
        "count": len(indices),
        "indices": indices.tolist(),
        "gold_labels": gold,
        "questions": QUESTIONS,
        "input_sha256": digest([texts, QUESTIONS]),
        "results": [],
        "limitation": "Small English classification sample. The original project includes AG News in its training mix; this is not an unseen-task generalization benchmark.",
    }
    names = list(QUESTIONS["topic"]["criteria"])
    for name in MODELS:
        predictions = {}
        for backend in ("torch-mps-float32", "mlx-float16"):
            agent = (
                load_reference(args.model_root / name)
                if backend.startswith("torch")
                else Agent(args.model_root / name)
            )
            predictions[backend] = [
                names.index(agent.predict(text, QUESTIONS)["answers"]["topic"]["choice"])
                for text in texts
            ]
            del agent
            gc.collect()
            torch.mps.empty_cache()
            mx.clear_cache()
        baseline, native = predictions.values()
        report["results"].append(
            {
                "model": name,
                "revision": MODELS[name],
                "predictions": predictions,
                "upstream_accuracy": float(np.mean(np.array(baseline) == gold)),
                "mlx_accuracy": float(np.mean(np.array(native) == gold)),
                "port_agreements": sum(a == b for a, b in zip(baseline, native)),
            }
        )
        save_json(args.output, report)
        summary = {k: v for k, v in report["results"][-1].items() if k != "predictions"}
        print(f"{name}: {summary}", flush=True)


if __name__ == "__main__":
    main()
