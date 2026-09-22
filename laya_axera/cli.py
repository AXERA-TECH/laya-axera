"""Command-line prediction and the web demo server."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def discover_checkpoints(root: Path):
    """Checkpoint subdirectories of an AXERA-TECH/Laya package, in a stable order."""
    found = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "config.json").is_file():
            config = json.loads((child / "config.json").read_text(encoding="utf-8"))
            axmodel = child / config.get("filename_axmodel", "model.axmodel")
            if axmodel.is_file():
                found[child.name] = child
    return found


def cmd_run(args):
    from .agent import Agent

    agent = Agent(args.model_dir, device_id=args.device, provider=args.provider)
    if args.input is not None:
        request = json.loads(args.input.read_text(encoding="utf-8"))
        print(json.dumps(agent.predict_request(request), indent=2, ensure_ascii=False))
        return
    # Resident JSON Lines mode: one request per line, /exit stops.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "/exit":
            break
        result = agent.predict_request(json.loads(line))
        print(json.dumps(result, ensure_ascii=False), flush=True)


def cmd_serve(args):
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("The web demo needs fastapi and uvicorn: pip install 'laya-axera[web]'")

    from .server import create_app

    checkpoints = {}
    if args.root is not None:
        checkpoints.update(discover_checkpoints(args.root))
    for spec in args.model or []:
        name, _, path = spec.rpartition("=")
        path = Path(path)
        checkpoints[name or path.name] = path
    if not checkpoints:
        raise SystemExit("No checkpoints. Pass --root <Laya package dir> or --model [name=]<dir>.")
    app = create_app(checkpoints, device_id=args.device, provider=args.provider)
    print(f"Serving {list(checkpoints)} on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="laya-axera", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Evaluate one request file, or JSONL on stdin")
    run.add_argument("model_dir", type=Path, help="Checkpoint directory, e.g. multilingual")
    run.add_argument("--input", type=Path, help="Request JSON file; omit for stdin JSONL mode")
    run.add_argument("--device", type=int, default=0, help="AXCL card index (default 0)")
    run.add_argument("--provider", help="Force an execution provider")
    run.set_defaults(func=cmd_run)

    serve = commands.add_parser("serve", help="Web demo: decision playground + Snake")
    serve.add_argument("--root", type=Path, help="AXERA-TECH/Laya package directory")
    serve.add_argument("--model", action="append", metavar="[NAME=]DIR", help="Add a checkpoint")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8010)
    serve.add_argument("--device", type=int, default=0, help="AXCL card index (default 0)")
    serve.add_argument("--provider", help="Force an execution provider")
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)
