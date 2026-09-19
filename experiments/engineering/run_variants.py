"""Isolated inference experiments; production runtime and checkpoints stay untouched.

Run from the repository root with ``python -m experiments.engineering.run_variants``.
All questions within a batch have distinct instructions and token sequences.
"""

import argparse
import gc
import hashlib
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from benchmarks.common import distribution, environment, parity_cases, save_json, workload
from laya_mlx import Agent
from laya_mlx.agent import collate_items

TOPICS = [
    "Does this message concern an invoice?",
    "Does the customer explicitly request a refund?",
    "Is there evidence of duplicate charging?",
    "Does the message mention a software crash?",
    "Does the customer need a response today?",
    "Is the customer asking about a new contract?",
    "Does the request concern an existing purchase?",
    "Is a monetary amount explicitly specified?",
    "Does the customer provide an invoice identifier?",
    "Does the message request technical troubleshooting?",
    "Is the customer reporting an authentication problem?",
    "Does the message describe an account cancellation?",
    "Is the request related to payment reconciliation?",
    "Does the customer ask for a status update?",
    "Is the sender offering a product for sale?",
    "Does the message ask for escalation to a human?",
    "Is there an explicit deadline in the message?",
    "Does the customer mention a delivery address?",
    "Is the sender requesting a pricing quotation?",
    "Does the request mention an attachment?",
    "Is a charge disputed by the customer?",
    "Does the customer mention an order number?",
    "Is the message a compliment about service?",
    "Does the customer report a security incident?",
    "Is the sender asking to change a subscription?",
    "Does the customer request a payment receipt?",
    "Does the message require action by the billing team?",
    "Is the customer asking about business opening hours?",
    "Does the sender mention another unresolved ticket?",
    "Is the request about a damaged physical product?",
    "Does the sender want to update contact details?",
    "Is the customer threatening legal action?",
    "Does the customer request an apology?",
    "Is a bank transfer mentioned in the message?",
    "Does the request involve multiple transactions?",
    "Is the customer trying to arrange a meeting?",
    "Does the request mention a credit card?",
    "Is the sender asking for documentation?",
    "Does the customer ask to reverse a transaction?",
    "Is the issue already described as resolved?",
    "Does the message contain a request for compensation?",
    "Is the sender acting on behalf of another person?",
    "Does the customer request an exchange of goods?",
    "Is the message a notification with no requested action?",
    "Does the customer state that a payment was successful?",
    "Does the sender report an unexpected recurring charge?",
    "Is the message related to tax documentation?",
    "Does the customer ask to speak with a manager?",
    "Is a service outage described in the message?",
    "Does the message ask for confirmation after resolution?",
]


def distinct_workload(count, long=False):
    state, _ = workload(1, long=long)
    questions = {
        f"q{i}": {
            "type": "choice",
            "instructions": instruction,
            "criteria": {
                "yes": "supported",
                "no": "not supported",
                "unclear": "insufficient evidence",
            },
        }
        for i, instruction in enumerate(TOPICS[:count])
    }
    return state, questions


class ExperimentalAgent(Agent):
    def forward(self, batch):
        with mx.stream(self.device):
            tensors = {key: mx.array(value) for key, value in batch.items()}
            output = self.infer(**tensors)
            mx.eval(output)
        return output


class CompiledLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.attention_type = getattr(layer, "attention_type", None)
        self._function = mx.compile(lambda x, mask: layer(x, mask))

    def __call__(self, x, mask):
        return self._function(x, mask)


def selected_head(
    model, input_ids, attention_mask, marker_pos, marker_mask, qtype, *, full_attention=False
):
    """Dependency-preserving pruning of unused final-head query/FFN outputs."""
    h = model.encoder(input_ids, attention_mask)
    h = h + model.type_emb(qtype)[:, None, :]
    mask = attention_mask[:, None, None, :].astype(mx.bool_)
    for layer in model.head.layers[:-1]:
        h = layer(h, mask)
    layer = model.head.layers[-1]
    attn = layer.self_attn
    b, length, _ = h.shape
    rows = mx.arange(b)[:, None]
    positions = mx.concatenate([mx.zeros((b, 1), mx.int32), mx.maximum(marker_pos, 0)], axis=1)
    qkv = attn.in_proj(layer.norm1(h)).reshape(b, length, 3, attn.num_heads, attn.head_dim)
    q = (qkv[:, :, 0] if full_attention else qkv[rows, positions, 0]).transpose(0, 2, 1, 3)
    k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in (1, 2)]
    attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.head_dim**-0.5, mask=mask)
    attended = attended.transpose(0, 2, 1, 3).reshape(
        b, length if full_attention else positions.shape[1], -1
    )
    if full_attention:
        attended = attended[rows, positions]
    selected = h[rows, positions] + attn.out_proj(attended)
    selected = selected + layer.linear2(nn.relu(layer.linear1(layer.norm2(selected))))
    logits = model.scorer(selected[:, 1:]).squeeze(-1).astype(mx.float32)
    logits = mx.where(marker_mask, logits, -1e4)
    p = mx.softmax(logits, axis=-1)
    k = mx.maximum(marker_mask.sum(axis=-1), 2).astype(mx.float32)
    entropy = -(p * mx.log(mx.maximum(p, 1e-9))).sum(axis=-1) / mx.log(k)
    top = mx.sort(p, axis=-1)[:, -2:]
    features = mx.stack([top[:, 1], top[:, 1] - top[:, 0], entropy, k / 255.0], axis=-1)
    pooled = mx.concatenate([selected[:, 0].astype(mx.float32), features], axis=-1)
    return logits, model.act_head(pooled.astype(model.act_head.layers[0].weight.dtype)).astype(
        mx.float32
    )


def measured(function, count):
    samples = []
    for _ in range(count):
        mx.synchronize()
        start = time.perf_counter_ns()
        output = function()
        mx.eval(output)
        mx.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "samples_ms": samples,
        "p50_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "mean_ms": float(np.mean(samples)),
    }


def cpu_outputs(function, tensors):
    output = function(**tensors)
    mx.eval(output)
    return tuple(np.asarray(value).copy() for value in output)


def compare(agent, actual, expected, items):
    probabilities = distribution(agent, actual[0], items)
    reference = distribution(agent, expected[0], items)
    return {
        "finite": bool(all(np.isfinite(v).all() for v in actual)),
        "max_logit_error": float(np.max(np.abs(actual[0] - expected[0]))),
        "max_action_logit_error": float(np.max(np.abs(actual[1] - expected[1]))),
        "max_probability_error": max(
            float(np.max(np.abs(a - b))) for a, b in zip(probabilities, reference)
        ),
        "argmax_agree": sum(
            int(np.argmax(a) == np.argmax(b)) for a, b in zip(probabilities, reference)
        ),
        "questions": len(items),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="laya", choices=["laya", "laya-multilingual", "laya-typed-decisions"]
    )
    parser.add_argument(
        "--variant",
        default="eager",
        choices=[
            "eager",
            "compiled",
            "blocks",
            "q8",
            "q4",
            "selected",
            "selected-compiled",
            "selected-full-attention",
            "metal-compiled",
        ],
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cases", default="short1,short16,long1")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "environment": environment(),
        "model": args.model,
        "variant": args.variant,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "measurement": "synchronized per-call wall time; fixed frozen weights; no result cache or deduplication",
        "results": [],
    }
    report["experiment_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    agent = ExperimentalAgent(Path("models") / args.model, dtype="float16", batch_size=64)
    agent.infer = agent.model
    prepared = []
    for case in args.cases.split(","):
        is_long = case.startswith("long")
        count = int(case.removeprefix("long" if is_long else "short"))
        state, questions = distinct_workload(count, long=is_long)
        items, _ = agent.prepare(state, questions)
        batch = collate_items(items, agent.tok.pad_token_id)
        tensors = {k: mx.array(v) for k, v in batch.items()}
        mx.eval(tensors)
        expected = cpu_outputs(agent.model, tensors)
        prepared.append((case, state, questions, items, batch, tensors, expected))
    quality = []
    if args.quality:
        for name, state, questions in parity_cases():
            items, _ = agent.prepare(state, questions)
            tensors = {
                k: mx.array(v) for k, v in collate_items(items, agent.tok.pad_token_id).items()
            }
            mx.eval(tensors)
            quality.append((name, items, tensors, cpu_outputs(agent.model, tensors)))
    if args.variant in ("q8", "q4"):
        nn.quantize(
            agent.model,
            group_size=64,
            bits=int(args.variant[1:]),
            class_predicate=lambda path, module: (
                path.startswith("encoder.layers.") and isinstance(module, nn.Linear)
            ),
        )
        mx.eval(agent.model.parameters())
        agent.infer = mx.compile(agent.model)
    elif args.variant == "compiled":
        agent.infer = mx.compile(agent.model)
    elif args.variant == "metal-compiled":
        from .kernels import MetalMLP

        for layer in agent.model.encoder.layers:
            layer.mlp = MetalMLP(layer.mlp)
        agent.infer = mx.compile(agent.model)
    elif args.variant == "blocks":
        agent.model.encoder.layers = [CompiledLayer(layer) for layer in agent.model.encoder.layers]
        agent.model.head.layers = [CompiledLayer(layer) for layer in agent.model.head.layers]
    elif args.variant.startswith("selected"):
        agent.infer = lambda **kwargs: selected_head(
            agent.model, full_attention=args.variant == "selected-full-attention", **kwargs
        )
        if args.variant.endswith("compiled") or args.variant == "selected-full-attention":
            agent.infer = mx.compile(agent.infer)
    gc.collect()
    mx.clear_cache()
    report["weight_bytes"] = sum(v.nbytes for _, v in tree_flatten(agent.model.parameters()))
    for case, state, questions, items, batch, tensors, expected in prepared:
        mx.clear_cache()
        mx.reset_peak_memory()
        cold = measured(lambda: agent.infer(**tensors), 1)
        for _ in range(args.warmup):
            mx.eval(agent.infer(**tensors))
            mx.synchronize()
        forward = measured(lambda: agent.infer(**tensors), args.iterations)
        forward_memory = {
            "active_bytes": mx.get_active_memory(),
            "peak_bytes": mx.get_peak_memory(),
            "cache_bytes": mx.get_cache_memory(),
        }
        for _ in range(args.warmup):
            agent.predict(state, questions)
            mx.synchronize()
        end_to_end = measured(lambda: agent.predict(state, questions), args.iterations)
        prep_start = time.perf_counter_ns()
        for _ in range(30):
            agent.prepare(state, questions)
        prep_ms = (time.perf_counter_ns() - prep_start) / 30e6
        actual = cpu_outputs(agent.infer, tensors)
        row = {
            "case": case,
            "batch": len(items),
            "length": int(tensors["input_ids"].shape[1]),
            "unique_inputs": len({tuple(item["ids"]) for item in items}),
            "useful_tokens": sum(len(item["ids"]) for item in items),
            "input_sha256": hashlib.sha256(batch["input_ids"].tobytes()).hexdigest(),
            "cold_forward": cold,
            "forward": forward,
            "end_to_end": end_to_end,
            "prepare_mean_ms": prep_ms,
            "memory": forward_memory,
            "parity": compare(agent, actual, expected, items),
        }
        report["results"].append(row)
        save_json(args.output, report)
        print(
            f"{args.model} {args.variant} {case} B{row['batch']} L{row['length']} unique={row['unique_inputs']}: forward={forward['p50_ms']:.3f} ms e2e={end_to_end['p50_ms']:.3f} ms cold={cold['p50_ms']:.2f} ms delta-p={row['parity']['max_probability_error']:.6f}",
            flush=True,
        )
    if quality:
        report["quality"] = []
        for name, items, tensors, expected in quality:
            report["quality"].append(
                {"case": name, **compare(agent, cpu_outputs(agent.infer, tensors), expected, items)}
            )
        save_json(args.output, report)
        print(
            "quality",
            sum(r["argmax_agree"] for r in report["quality"]),
            "/",
            sum(r["questions"] for r in report["quality"]),
            "max delta-p",
            max(r["max_probability_error"] for r in report["quality"]),
            flush=True,
        )


if __name__ == "__main__":
    main()
