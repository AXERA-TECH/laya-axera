"""Download exactly the checkpoints used in the checked-in benchmark."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from .common import MODELS, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()
    for name, revision in MODELS.items():
        snapshot_download(
            "convaiinnovations/" + name,
            revision=revision,
            local_dir=args.output / name,
            allow_patterns=[
                "model.safetensors",
                "rl_agent_config.json",
                "encoder/config.json",
                "tokenizer/*",
            ],
        )
    save_json(args.output / "revisions.json", MODELS)


if __name__ == "__main__":
    main()
