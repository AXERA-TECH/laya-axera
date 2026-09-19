"""One backend/checkpoint per fresh process; called by benchmarks.run."""

import argparse
import gc
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import MODELS, digest, environment, load_reference, save_json, workload


def measure(function, synchronize, warmup, iterations):
    for _ in range(warmup):
        function()
        synchronize()
    samples = []
    for _ in range(iterations):
        synchronize()
        start = time.perf_counter_ns()
        function()
        synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "samples_ms": samples,
        "p50_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "mean_ms": float(np.mean(samples)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--backend", choices=("mlx", "torch-mps"), required=True)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1:
        parser.error("iterations and warmup must be positive")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment(),
        "model": args.model,
        "revision": MODELS[args.model],
        "backend": args.backend,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "batch_size_limit": args.batch_size if args.backend == "mlx" else None,
        "timing": "wall clock, synchronized GPU completion; downloads and model loading excluded",
        "results": [],
    }
    model_path = args.model_root / args.model
    load_start = time.perf_counter()
    if args.backend == "mlx":
        import mlx.core as mx
        from mlx.utils import tree_flatten

        from laya_mlx import Agent
        from laya_mlx.agent import collate_items

        agent = Agent(model_path, dtype=args.dtype, batch_size=args.batch_size)
        synchronize = mx.synchronize
        report["device"] = mx.device_info()
        report["parameter_count"] = (
            sum(v.size for _, v in tree_flatten(agent.model.parameters())) - 3
        )
        report["weight_bytes"] = sum(v.nbytes for _, v in tree_flatten(agent.model.parameters()))
    else:
        if args.dtype != "float32":
            parser.error("The stock upstream MPS runtime uses float32")
        import torch

        agent = load_reference(model_path)
        from laya.common import QTYPES, build_sequence, collate_items

        synchronize = torch.mps.synchronize
        report["device"] = {"device_name": report["environment"]["cpu"], "type": str(agent.device)}
        report["torch_threads"] = torch.get_num_threads()
        report["parameter_count"] = sum(p.numel() for p in agent.model.parameters())
        report["weight_bytes"] = sum(p.numel() * p.element_size() for p in agent.model.parameters())
    synchronize()
    report["load_seconds"] = time.perf_counter() - load_start
    for long, count in [(False, n) for n in (1, 5, 10, 50)] + [(True, n) for n in (1, 10)]:
        state, questions = workload(count, long=long)
        result = agent.predict(state, questions)
        if args.backend == "mlx":
            items, _ = agent.prepare(state, questions)
            batch = collate_items(items, agent.tok.pad_token_id)
            batch = {k: mx.array(v) for k, v in batch.items()}
            mx.eval(batch)

            def forward(batch=batch):
                outputs = agent.model(**batch)
                mx.eval(outputs)

            mx.clear_cache()
            mx.reset_peak_memory()
        else:
            items = []
            for definition in questions.values():
                q = agent._to_internal(definition)
                ids, markers = build_sequence(
                    agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"]
                )
                items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            raw = collate_items([items], agent.tok.pad_token_id)
            batch = {
                k: raw[k].to(agent.device)
                for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
            }

            @torch.inference_mode()
            def forward(batch=batch):
                agent.model(**batch)

        forward_times = measure(forward, synchronize, args.warmup, args.iterations)
        e2e = measure(
            lambda: agent.predict(state, questions), synchronize, args.warmup, args.iterations
        )
        e2e["questions_per_second"] = count * 1000 / e2e["mean_ms"]
        e2e["tokens_per_second"] = result["usage"]["input_tokens"] * 1000 / e2e["mean_ms"]
        memory = (
            {
                "mlx_active_bytes": mx.get_active_memory(),
                "mlx_peak_bytes": mx.get_peak_memory(),
                "mlx_cache_bytes": mx.get_cache_memory(),
            }
            if args.backend == "mlx"
            else {
                "mps_current_tensor_bytes": torch.mps.current_allocated_memory(),
                "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
                "note": "MPS exposes current allocations here, not a peak comparable to MLX peak memory",
            }
        )
        row = {
            "workload": "long" if long else "short",
            "questions": count,
            "sequence_length": batch["input_ids"].shape[1],
            "input_tokens": result["usage"]["input_tokens"],
            "input_sha256": digest([state, questions]),
            "output_sha256": digest(result),
            "forward": forward_times,
            "end_to_end": e2e,
            "memory": memory,
        }
        report["results"].append(row)
        save_json(args.output, report)
        print(
            f"{args.model} {args.backend}/{args.dtype} {row['workload']} q={count}: p50={e2e['p50_ms']:.2f} ms, p95={e2e['p95_ms']:.2f} ms, {e2e['questions_per_second']:.1f} q/s",
            flush=True,
        )
        del forward, batch
        gc.collect()


if __name__ == "__main__":
    main()
