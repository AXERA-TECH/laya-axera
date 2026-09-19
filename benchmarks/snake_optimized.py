"""Paired full-loop check of the actual shipped eager and opt-in optimized paths."""

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx

from laya_mlx.snake.benchmark import run_episode
from laya_mlx.snake.game import SnakeGame
from laya_mlx.snake.policy import LayaPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/snake/optimized-paired.json")
    )
    args = parser.parse_args()
    policies = {"eager": LayaPolicy(), "optimized": LayaPolicy(optimize=True)}
    for policy in policies.values():
        game = SnakeGame(seed=101)
        for _ in range(30):
            game.step(policy.decide(game).executed)
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    report = {
        "method": "Four paired seeds, alternating policy order; shipped public API paths, same game and truecolor rendering. All steps perform fresh predictions. Rendering excludes terminal emulator painting.",
        "models": {k: p.metadata for k, p in policies.items()},
        "episodes": {k: [] for k in policies},
        "active_memory_before": before,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, seed in enumerate((101, 102, 103, 104)):
        order = list(policies) if index % 2 == 0 else list(reversed(policies))
        for name in order:
            print(name, flush=True)
            report["episodes"][name].append(
                run_episode(policies[name], seed=seed, steps=args.steps, width=24, height=16)
            )
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    gc.collect()
    mx.clear_cache()
    report["active_memory_after"] = mx.get_active_memory()
    report["active_memory_growth"] = report["active_memory_after"] - before
    report["summary"] = {}
    for name, episodes in report["episodes"].items():
        report["summary"][name] = {
            "steps": sum(e["steps"] for e in episodes),
            "steps_per_second": sum(e["steps"] for e in episodes)
            / sum(e["seconds"] for e in episodes),
            "deaths": sum(not e["alive"] for e in episodes),
            "interventions": sum(e["interventions"] for e in episodes),
        }
    report["executed_agreement"] = sum(
        a["executed"] == b["executed"]
        for first, second in zip(report["episodes"]["eager"], report["episodes"]["optimized"])
        for a, b in zip(first["trace"], second["trace"])
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "summary": report["summary"],
                "active_memory_growth": report["active_memory_growth"],
                "executed_agreement": report["executed_agreement"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
