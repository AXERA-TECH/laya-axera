"""Benchmark a checkpoint: python examples/bench.py <model_dir> [-n 20] [--device N]"""

import argparse
import statistics
import time

import laya_axera as laya


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("-n", type=int, default=20, help="measured request iterations")
    parser.add_argument("--device", type=int, default=0, help="AXCL card index")
    args = parser.parse_args()

    started = time.perf_counter()
    agent = laya.load(args.model_dir, device_id=args.device)
    load_s = time.perf_counter() - started
    print(f"model={agent.model_name} provider={agent.provider} load={load_s:.1f}s")

    request = agent.sample_request() or {
        "state": "Invoice 4411 was charged twice. Please refund the duplicate today.",
        "questions": {
            "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"}
        },
    }
    agent.predict_request(request)  # warmup
    latencies = []
    wall = time.perf_counter()
    for _ in range(args.n):
        result = agent.predict_request(request)
        latencies += [a["npu_latency_ms"] for a in result["answers"].values()]
    wall = time.perf_counter() - wall
    latencies.sort()
    n = len(latencies)
    print(
        f"questions={n} mean={statistics.mean(latencies):.2f}ms"
        f" p50={latencies[n // 2]:.2f}ms min={latencies[0]:.2f}ms max={latencies[-1]:.2f}ms"
        f" throughput={n / wall:.1f} q/s"
    )


if __name__ == "__main__":
    main()
