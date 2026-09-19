"""CPU-only analysis of paired experiment files; preserves the raw timings."""

import json
from pathlib import Path

import numpy as np

from benchmarks.common import save_json


def main():
    rng = np.random.default_rng(20260919)
    report = {
        "method": "median of paired per-round eager/candidate ratios; percentile bootstrap of round indices (2000 resamples), exploratory 95% intervals",
        "results": [],
    }
    for path in sorted(Path("experiments/engineering").glob("*-paired.json")):
        raw = json.loads(path.read_text())
        for row in raw["results"]:
            for mode in ("forward", "end_to_end"):
                baseline = np.asarray(row["candidates"]["eager"][mode]["samples_ms"])
                for candidate, metrics in row["candidates"].items():
                    if candidate == "eager":
                        continue
                    times = np.asarray(metrics[mode]["samples_ms"])
                    ratios = baseline / times
                    resampled = ratios[rng.integers(len(ratios), size=(2000, len(ratios)))]
                    interval = np.percentile(np.median(resampled, axis=1), [2.5, 97.5])
                    report["results"].append(
                        {
                            "source": path.name,
                            "case": row["case"],
                            "candidate": candidate,
                            "mode": mode,
                            "rounds": len(ratios),
                            "paired_speedup_median": float(np.median(ratios)),
                            "bootstrap_95_interval": interval.tolist(),
                            "per_round_min_max": [float(ratios.min()), float(ratios.max())],
                            "ratio_of_p50": float(np.median(baseline) / np.median(times)),
                        }
                    )
    save_json(Path("experiments/engineering/paired_analysis.json"), report)


if __name__ == "__main__":
    main()
