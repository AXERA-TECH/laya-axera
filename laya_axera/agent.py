"""Laya typed-decision inference on AXERA NPUs through PyAXEngine.

The packaged AX8850 graphs are fixed-shape: batch 1, 256 tokens, up to 4 options,
one NPU forward per question. Prompt construction, calibration and the result
schema follow upstream Laya (see NOTICE).
"""

import contextlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .common import (
    QTYPES,
    build_sequence,
    confidence_from_probs,
    render_options,
    temp_bucket,
    to_internal,
)
from .tokenizer import Tokenizer

ONCHIP_PROVIDER = "AxEngineExecutionProvider"
AXCL_PROVIDER = "AXCLRTExecutionProvider"


def _resolve_model_dir(model_dir) -> Path:
    path = Path(model_dir).expanduser().resolve()
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Not a Laya AXERA checkpoint: {path}/config.json is missing")
    return path


class Agent:
    """One resident Laya checkpoint on one NPU.

    `model_dir` is a checkpoint directory from the AXERA-TECH/Laya package
    (`english/`, `multilingual/` or `typed-decisions/`), containing
    `config.json`, `model.axmodel` and `tokenizer/`.

    The provider is selected automatically: the on-chip runtime on an AX8850
    board, or the AXCL runtime on a host with PCIe/M.2 cards (`device_id`
    picks the card). Pass `provider` to force one.
    """

    def __init__(self, model_dir, *, device_id=0, provider=None):
        # PyAXEngine prints its diagnostics with bare print(); keep stdout clean
        # for JSON consumers by routing them to stderr during import and load.
        with contextlib.redirect_stdout(sys.stderr):
            try:
                import axengine
            except ImportError as exc:
                raise RuntimeError(
                    "PyAXEngine is missing. Install an axengine wheel from "
                    "https://github.com/AXERA-TECH/pyaxengine/releases"
                ) from exc

        self.model_dir = _resolve_model_dir(model_dir)
        self.cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.seq_len = int(self.cfg.get("sequence_length", 256))
        self.num_options = int(self.cfg.get("num_options", 4))
        self.head_max_len = int(self.cfg.get("head_max_len", 128))
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.model_name = self.cfg.get("model_name", self.model_dir.name)
        self.tok = Tokenizer(self.model_dir / self.cfg.get("tokenizer_dir", "tokenizer"))

        available = axengine.get_available_providers()
        if provider is None:
            for candidate in (ONCHIP_PROVIDER, AXCL_PROVIDER):
                if candidate in available:
                    provider = candidate
                    break
            else:
                raise RuntimeError(
                    f"No usable NPU provider; PyAXEngine reports {available}. "
                    "Run on an AX8850 board or on a host with AXCL cards."
                )
        elif provider not in available:
            raise RuntimeError(f"Provider {provider!r} is unavailable; found {available}")
        self.provider = provider
        self.device_id = int(device_id)

        model_path = self.model_dir / self.cfg.get("filename_axmodel", "model.axmodel")
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        # AXCLRT reads the card index from the provider_options argument
        # (a list of dicts), not from a (name, options) provider tuple.
        options = [{"device_id": self.device_id}] if provider == AXCL_PROVIDER else None
        with contextlib.redirect_stdout(sys.stderr):
            self._session = axengine.InferenceSession(
                str(model_path), providers=[provider], provider_options=options
            )
        # One graph execution at a time per session; serve handlers share this Agent.
        self._lock = threading.Lock()

    def encode_question(self, state, qdef):
        """Build the fixed-shape int32 feed for one question."""
        q = to_internal(qdef)
        options = render_options(q)
        if not 2 <= len(options) <= self.num_options:
            raise ValueError(
                f"question has {len(options)} options; this graph supports 2..{self.num_options}"
            )
        ids, markers = build_sequence(self.tok, state, q, self.seq_len, self.head_max_len)
        if len(markers) != len(options):
            raise ValueError("an option marker was truncated; shorten the question")
        pad = self.seq_len - len(ids)
        feed = {
            "input_ids": np.asarray([ids + [self.tok.pad_token_id] * pad], dtype=np.int32),
            "attention_mask": np.asarray([[1] * len(ids) + [0] * pad], dtype=np.int32),
            "marker_pos": np.asarray(
                [markers + [0] * (self.num_options - len(markers))], dtype=np.int32
            ),
            "marker_mask": np.asarray(
                [[1] * len(markers) + [0] * (self.num_options - len(markers))], dtype=np.int32
            ),
            "qtype": np.asarray([QTYPES[q["t"]]], dtype=np.int32),
        }
        return feed, q, len(ids)

    def _forward(self, feed):
        # Timed inside the lock: queueing behind concurrent callers is not NPU time.
        with self._lock:
            started = time.perf_counter()
            logits, act_logits = self._session.run(None, feed)
            latency_ms = (time.perf_counter() - started) * 1000.0
        return np.asarray(logits), np.asarray(act_logits), latency_ms

    def system_one(self, state, questions):
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty dictionary keyed by question id")
        answers = {}
        total_ms = 0.0
        input_tokens = 0
        for qid, qdef in questions.items():
            feed, q, length = self.encode_question(state, qdef)
            input_tokens += length
            logits, act, latency_ms = self._forward(feed)
            total_ms += latency_ms
            if not np.isfinite(logits).all() or not np.isfinite(act).all():
                raise FloatingPointError(f"Non-finite model outputs for question {qid!r}")
            k, qt = int(feed["marker_mask"].sum()), QTYPES[q["t"]]
            scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            z = logits[0, :k].astype(np.float64) / max(1e-3, float(scale))
            p = np.exp(z - z.max())
            p /= p.sum()
            a = act[0].astype(np.float64)
            a = np.exp(a - a.max())
            a /= a.sum()
            answer = {
                "type": q["t"],
                "confidence": round(confidence_from_probs(p, k), 4),
                "action": {"act_probability": round(float(a[0]), 4)},
                "npu_latency_ms": round(latency_ms, 3),
            }
            if q["t"] == "choice":
                labels = list(q["crit"])
                answer.update(
                    choice=labels[int(p.argmax())],
                    probabilities={label: round(float(v), 4) for label, v in zip(labels, p)},
                )
            elif q["t"] == "score":
                answer.update(
                    score=round(float((np.arange(k) * p).sum()), 4),
                    legend={str(i): value for i, value in enumerate(q["crit"])},
                    probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)},
                )
            else:
                answer.update(
                    noul=round(float(p[1]), 4),
                    confidence=round(max(float(p[1]), 1.0 - float(p[1])), 4),
                )
            answers[qid] = answer
        return {
            "model": self.model_name,
            "engine": {"provider": self.provider, "device_id": self.device_id},
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "total_npu_latency_ms": round(total_ms, 3),
        }

    predict = system_one

    def predict_request(self, request):
        """Evaluate a full request object, the axllm `{state, questions}` schema."""
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        if "state" not in request:
            raise ValueError("request is missing required field: state")
        return self.predict(request["state"], request.get("questions"))

    def sample_request(self):
        """The board-validated sample request shipped beside the checkpoint, if any."""
        path = self.model_dir / "sample_request.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return None


def load(model_dir, *, device_id=0, provider=None):
    return Agent(model_dir, device_id=device_id, provider=provider)
