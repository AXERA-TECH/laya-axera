"""Compare two real checkpoints across paired, fixed-horizon Snake episodes."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from laya_mlx.snake.benchmark import run_episode
from laya_mlx.snake.game import SnakeGame
from laya_mlx.snake.policy import LayaPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/snake/model-comparison.json")
    )
    args = parser.parse_args()
    if min(args.episodes, args.steps) < 1:
        parser.error("Episode and step counts must be positive")
    policies = {
        "laya": LayaPolicy("models/hub/laya-mlx"),
        "multilingual": LayaPolicy("models/hub/laya-multilingual-mlx"),
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Two checkpoints, identical episode seeds and horizon. Alternate checkpoint order per seed. Both use planner features and cycle shield. Headless: UI rendering excluded; each action requires fresh inference. Scores are measured at the fixed horizon, not at death or board completion.",
        "settings": {"episodes_per_model": args.episodes, "steps_per_episode": args.steps},
        "models": {k: p.metadata for k, p in policies.items()},
        "episodes": {k: [] for k in policies},
    }
    for policy in policies.values():
        warm = SnakeGame(seed=5000)
        for _ in range(10):
            warm.step(policy.decide(warm).executed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.episodes):
        names = list(policies) if index % 2 == 0 else list(reversed(policies))
        for name in names:
            print(name, flush=True)
            report["episodes"][name].append(
                run_episode(
                    policies[name],
                    seed=2000 + index,
                    steps=args.steps,
                    width=24,
                    height=16,
                    render=False,
                )
            )
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["summary"] = {}
    for name, episodes in report["episodes"].items():
        timings = [t["inference_ms"] for e in episodes for t in e["trace"]]
        scores = [e["score"] for e in episodes]
        report["summary"][name] = {
            "episodes": len(episodes),
            "steps": sum(e["steps"] for e in episodes),
            "survived_episodes": sum(e["alive"] for e in episodes),
            "median_score": float(np.median(scores)),
            "mean_score": float(np.mean(scores)),
            "inference_p50_ms": float(np.percentile(timings, 50)),
            "inference_p95_ms": float(np.percentile(timings, 95)),
            "interventions": sum(e["interventions"] for e in episodes),
        }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
