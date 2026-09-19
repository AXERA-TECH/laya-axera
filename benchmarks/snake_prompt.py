"""Compare prompt costs in alternating order on identical, recorded game states."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from laya_mlx.snake.game import SnakeGame
from laya_mlx.snake.policy import LayaPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/snake/prompt-comparison.json")
    )
    args = parser.parse_args()
    if not 0 <= args.warmup < args.steps:
        parser.error("Require 0 <= warmup < steps")
    policy = LayaPolicy(args.model)
    game = SnakeGame(seed=args.seed)
    records = []
    for index in range(args.steps):
        record = {"board": game.snapshot(), "decisions": {}}
        order = ["compact", "detailed"] if index % 2 == 0 else ["detailed", "compact"]
        record["order"] = order
        for prompt in order:
            policy.prompt = prompt
            record["decisions"][prompt] = policy.decide(game).to_dict()
        records.append(record)
        game.step(record["decisions"]["compact"]["executed"])
        if game.won or not game.alive:
            break
    summary = {}
    for prompt in ("compact", "detailed"):
        values = [r["decisions"][prompt] for r in records]
        timings = [d["inference_ms"] for d in values[args.warmup :]]
        summary[prompt] = {
            "timed_samples": len(timings),
            "median_inference_ms": float(np.median(timings)),
            "mean_input_tokens_all_states": float(np.mean([d["input_tokens"] for d in values])),
            "interventions_all_states": sum(d["intervened"] for d in values),
        }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Alternate prompt order at each identical board. Game follows compact. Exclude first warmup iterations from timing summaries; include every record.",
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": policy.metadata,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
