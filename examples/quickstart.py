"""Run with: python examples/quickstart.py /path/to/AXERA-TECH/Laya/english"""

import json
import sys
from pathlib import Path

import laya_axera as laya

root = Path(__file__).parent
agent = laya.load(sys.argv[1] if len(sys.argv) > 1 else "models/Laya/english")
result = agent.predict(
    json.loads((root / "state.json").read_text()),
    json.loads((root / "questions.json").read_text()),
)
print(json.dumps(result, indent=2, ensure_ascii=False))
