"""Compare real checkpoint outputs with upstream, then exercise repeated inference."""

import argparse
import gc
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from laya_mlx import Agent
from laya_mlx.agent import collate_items

from .common import (
    MODELS,
    digest,
    distribution,
    environment,
    load_reference,
    parity_cases,
    save_json,
    softmax,
)


def validate_model(name, root, repeats, optimize=False):
    import torch
    from laya.common import QTYPES, build_sequence
    from laya.common import collate_items as torch_collate

    reference = load_reference(root / name)
    prepared = []
    for case_name, state, questions in parity_cases():
        ref_items = []
        for q in questions.values():
            internal = reference._to_internal(q)
            ids, markers = build_sequence(
                reference.tok,
                state,
                internal,
                reference.cfg["max_len"],
                reference.cfg["head_max_len"],
            )
            ref_items.append({"ids": ids, "markers": markers, "qtype": QTYPES[internal["t"]]})
        batch = torch_collate([ref_items], reference.tok.pad_token_id)
        with torch.inference_mode():
            logits, act = reference.model(
                **{
                    k: batch[k].to(reference.device)
                    for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
                }
            )
            torch.mps.synchronize()
        logits, act = logits.cpu().numpy(), act.cpu().numpy()
        prepared.append(
            {
                "name": case_name,
                "state": state,
                "questions": questions,
                "items": ref_items,
                "logits": logits,
                "act": act,
                "result": reference.predict(state, questions),
                "probabilities": distribution(reference, logits, ref_items),
            }
        )
    del reference
    gc.collect()
    torch.mps.empty_cache()
    reports = []
    for dtype in ("float32", "float16"):
        agent = Agent(
            root / name,
            dtype=dtype,
            compile=optimize,
            pad_to_multiple=16 if optimize else None,
            cache_prompts=optimize,
        )
        cases = []
        max_probability_error = 0.0
        agree = total = 0
        expected = {}
        for entry in prepared:
            items, _ = agent.prepare(entry["state"], entry["questions"])
            assert items == entry["items"], f"Tokenizer mismatch: {name}/{entry['name']}"
            batch = collate_items(
                items,
                agent.tok.pad_token_id,
                pad_to_multiple=agent.pad_to_multiple,
                max_length=agent.cfg.get("max_len", 512),
            )
            logits, act = agent.forward(batch)
            logits, act = np.asarray(logits), np.asarray(act)
            assert np.isfinite(logits).all() and np.isfinite(act).all()
            probabilities = distribution(agent, logits, items)
            error = max(
                float(np.max(np.abs(a - b))) for a, b in zip(probabilities, entry["probabilities"])
            )
            agreement = sum(
                int(a.argmax() == b.argmax()) for a, b in zip(probabilities, entry["probabilities"])
            )
            action_error = float(np.max(np.abs(softmax(act) - softmax(entry["act"]))))
            result = agent.predict(entry["state"], entry["questions"])
            assert result["usage"] == entry["result"]["usage"]
            expected[entry["name"]] = digest(result)
            cases.append(
                {
                    "case": entry["name"],
                    "questions": len(items),
                    "sequence_length": batch["input_ids"].shape[1],
                    "input_sha256": digest([entry["state"], entry["questions"]]),
                    "tokens_equal": True,
                    "logits_max_abs_error": float(np.max(np.abs(logits - entry["logits"]))),
                    "action_logits_max_abs_error": float(np.max(np.abs(act - entry["act"]))),
                    "probability_max_abs_error": error,
                    "action_probability_max_abs_error": action_error,
                    "argmax_agreements": agreement,
                    "public_result_equal": result == entry["result"],
                }
            )
            max_probability_error = max(max_probability_error, error)
            agree += agreement
            total += len(items)
        tolerance = 1e-4 if dtype == "float32" else 0.02
        assert max_probability_error <= tolerance, (name, dtype, max_probability_error, tolerance)
        assert all(c["action_probability_max_abs_error"] <= tolerance for c in cases)
        gc.collect()
        mx.clear_cache()
        memory_before = mx.get_active_memory()
        mx.reset_peak_memory()
        start = time.perf_counter()
        for index in range(repeats):
            entry = prepared[index % len(prepared)]
            result = agent.predict(entry["state"], entry["questions"])
            assert digest(result) == expected[entry["name"]], f"Output drift on iteration {index}"
        elapsed = time.perf_counter() - start
        gc.collect()
        mx.clear_cache()
        memory_after = mx.get_active_memory()
        assert memory_after - memory_before < 32 * 1024**2, "Unexpected active memory growth"
        report = {
            "model": name,
            "revision": MODELS[name],
            "dtype": dtype,
            "optimized": optimize,
            "argmax_agreements": agree,
            "questions": total,
            "probability_max_abs_error": max_probability_error,
            "probability_tolerance": tolerance,
            "cases": cases,
            "stability": {
                "calls": repeats,
                "elapsed_seconds": elapsed,
                "finite": True,
                "deterministic": True,
                "active_bytes_before": memory_before,
                "active_bytes_after": memory_after,
                "active_growth_bytes": memory_after - memory_before,
                "peak_bytes": mx.get_peak_memory(),
            },
        }
        reports.append(report)
        print(
            f"{name} {dtype}: {agree}/{total} argmax agreement, max probability error {max_probability_error:.6g}, {repeats} stable calls",
            flush=True,
        )
        del agent
        gc.collect()
        mx.clear_cache()
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/validation.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    # Import the actual pinned upstream package before importing its prompt helpers.
    import sys

    from .common import ROOT

    sys.path.insert(0, str(ROOT / ".upstream"))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment(),
        "results": [],
    }
    for name in args.models:
        report["results"].extend(validate_model(name, args.model_root, args.repeats, args.optimize))
        save_json(args.output, report)


if __name__ == "__main__":
    main()
