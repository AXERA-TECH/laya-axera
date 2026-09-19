"""Interleaved, changing-input confirmation of exact engineering candidates."""

import argparse
import hashlib
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np

from benchmarks.common import environment, save_json
from laya_mlx.agent import collate_items

from .run_variants import ExperimentalAgent, compare, cpu_outputs, distinct_workload, selected_head


def stats(samples):
    return {
        "samples_ms": samples,
        "p50_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", default="short1,short16,long1,long8,short50")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument(
        "--metal", action="store_true", help="Compare custom FP16 GELU/gate instead of head pruning"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    agent = ExperimentalAgent(Path("models") / args.model, dtype="float16", batch_size=64)
    functions = {
        "eager": agent.model,
        "compiled": mx.compile(agent.model),
        "selected-compiled": mx.compile(lambda **kw: selected_head(agent.model, **kw)),
        "selected-full-attention": mx.compile(
            lambda **kw: selected_head(agent.model, full_attention=True, **kw)
        ),
    }
    if args.metal:
        from .kernels import MetalMLP

        other = ExperimentalAgent(Path("models") / args.model, dtype="float16", batch_size=64)
        for layer in other.model.encoder.layers:
            layer.mlp = MetalMLP(layer.mlp)
        functions = {
            "eager": agent.model,
            "compiled": mx.compile(agent.model),
            "metal": other.model,
            "metal-compiled": mx.compile(other.model),
        }
    report = {
        "environment": environment(),
        "model": args.model,
        "iterations": args.iterations,
        "method": "cyclic candidate order within each round, changing actual state every round, same prepared tensor inputs per candidate; no cache or deduplication",
        "results": [],
    }
    report["experiment_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for case in args.cases.split(","):
        is_long = case.startswith("long")
        count = int(case.removeprefix("long" if is_long else "short"))
        alternatives = []
        for index in range(16):
            state, questions = distinct_workload(count, long=is_long)
            state["body"] = f"Record {index}: " + state["body"]
            items, _ = agent.prepare(state, questions)
            batch = collate_items(items, agent.tok.pad_token_id)
            tensors = {key: mx.array(value) for key, value in batch.items()}
            mx.eval(tensors)
            alternatives.append((state, questions, items, tensors))
        shapes = Counter(tuple(alt[3]["input_ids"].shape) for alt in alternatives)
        shape = shapes.most_common(1)[0][0]
        alternatives = [alt for alt in alternatives if tuple(alt[3]["input_ids"].shape) == shape]
        assert len(alternatives) >= 2
        input_hashes = [
            hashlib.sha256(np.asarray(alt[3]["input_ids"]).tobytes()).hexdigest()
            for alt in alternatives
        ]
        assert len(set(input_hashes)) == len(alternatives)
        for function in functions.values():
            for _, _, _, tensors in alternatives[:5]:
                mx.eval(function(**tensors))
                mx.synchronize()
        samples = {kind: {"forward": [], "end_to_end": []} for kind in functions}
        names = list(functions)
        for mode in ("forward", "end_to_end"):
            for iteration in range(args.iterations):
                state, questions, items, tensors = alternatives[iteration % len(alternatives)]
                for offset in range(len(names)):
                    name = names[(iteration + offset) % len(names)]
                    agent.infer = functions[name]
                    mx.synchronize()
                    started = time.perf_counter_ns()
                    if mode == "forward":
                        output = agent.infer(**tensors)
                        mx.eval(output)
                    else:
                        agent.predict(state, questions)
                    mx.synchronize()
                    samples[name][mode].append((time.perf_counter_ns() - started) / 1e6)
        parity = {name: [] for name in functions if name != "eager"}
        for _, _, items, tensors in alternatives:
            reference = cpu_outputs(functions["eager"], tensors)
            for name in parity:
                parity[name].append(
                    compare(agent, cpu_outputs(functions[name], tensors), reference, items)
                )
        row = {
            "case": case,
            "shape": shape,
            "unique_questions_per_request": count,
            "distinct_input_batches": len(alternatives),
            "input_sha256": input_hashes,
            "candidates": {
                name: {mode: stats(values) for mode, values in modes.items()}
                for name, modes in samples.items()
            },
            "parity": parity,
        }
        report["results"].append(row)
        save_json(args.output, report)
        print(
            args.model,
            case,
            shape,
            {
                name: round(candidate["end_to_end"]["p50_ms"], 3)
                for name, candidate in row["candidates"].items()
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
