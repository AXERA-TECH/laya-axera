"""FastAPI web demo: a typed-decision playground and a Laya-driven Snake game."""

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .agent import Agent
from .flappy.game import FlappyGame
from .flappy.policy import LayaFlappyPolicy
from .snake.game import SnakeGame
from .snake.policy import LayaPolicy

WEB_DIR = Path(__file__).parent / "web"
MAX_SNAKE_SESSIONS = 16


class ModelRegistry:
    """Lazily loads one Agent per checkpoint; loading and inference are serialized."""

    def __init__(self, checkpoints: Dict[str, Path], device_id: int, provider: Optional[str]):
        if not checkpoints:
            raise ValueError("At least one checkpoint is required")
        self.checkpoints = {name: Path(path) for name, path in checkpoints.items()}
        self.device_id = device_id
        self.provider = provider
        # multilingual is the fastest checkpoint and covers Chinese; prefer it as default
        self.default = (
            "multilingual" if "multilingual" in self.checkpoints else next(iter(self.checkpoints))
        )
        self._agents: Dict[str, Agent] = {}
        self._errors: Dict[str, str] = {}
        self._loading: set = set()
        self._lock = threading.Lock()

    def status(self, name):
        if name in self._agents:
            return "ready"
        if name in self._loading:
            return "loading"
        if name in self._errors:
            return "error"
        return "unloaded"

    def get(self, name: Optional[str]) -> Agent:
        name = name or self.default
        if name not in self.checkpoints:
            raise HTTPException(404, f"Unknown model {name!r}; have {list(self.checkpoints)}")
        with self._lock:
            agent = self._agents.get(name)
            if agent is not None:
                return agent
            self._loading.add(name)
        try:
            agent = Agent(
                self.checkpoints[name], device_id=self.device_id, provider=self.provider
            )
        except Exception as exc:
            with self._lock:
                self._loading.discard(name)
                self._errors[name] = str(exc)
            raise HTTPException(500, f"Failed to load {name!r}: {exc}")
        with self._lock:
            self._loading.discard(name)
            self._errors.pop(name, None)
            self._agents[name] = agent
        return agent

    def describe(self):
        rows = []
        for name, path in self.checkpoints.items():
            row = {"name": name, "path": str(path), "status": self.status(name)}
            agent = self._agents.get(name)
            if agent is not None:
                row.update(
                    model_name=agent.model_name,
                    provider=agent.provider,
                    device_id=agent.device_id,
                    sequence_length=agent.seq_len,
                    num_options=agent.num_options,
                )
            if name in self._errors:
                row["error"] = self._errors[name]
            rows.append(row)
        return rows


class PredictBody(BaseModel):
    state: Any
    questions: Dict[str, Any]
    model: Optional[str] = None


class SnakeNewBody(BaseModel):
    model: Optional[str] = None
    width: int = 24
    height: int = 16
    seed: Optional[int] = None
    guarded: bool = True
    prompt: str = "compact"


class SnakeStepBody(BaseModel):
    session: str


class FlappyNewBody(BaseModel):
    model: Optional[str] = None
    seed: Optional[int] = None
    guarded: bool = True


def create_app(checkpoints: Dict[str, Path], *, device_id=0, provider=None) -> FastAPI:
    registry = ModelRegistry(checkpoints, device_id, provider)
    sessions: Dict[str, Dict[str, Any]] = {}
    sessions_lock = threading.Lock()

    app = FastAPI(title="laya-axera", version=__version__)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/api/info")
    def info():
        return {
            "version": __version__,
            "default_model": registry.default,
            "models": registry.describe(),
        }

    @app.get("/api/models")
    def models():
        return registry.describe()

    @app.get("/api/samples/{name}")
    def sample(name: str):
        if name not in registry.checkpoints:
            raise HTTPException(404, f"Unknown model {name!r}")
        path = registry.checkpoints[name] / "sample_request.json"
        if not path.is_file():
            raise HTTPException(404, f"{name} ships no sample_request.json")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/api/predict")
    def predict(body: PredictBody):
        agent = registry.get(body.model)
        try:
            return agent.predict(body.state, body.questions)
        except (ValueError, FloatingPointError) as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/snake/new")
    def snake_new(body: SnakeNewBody):
        agent = registry.get(body.model)
        seed = body.seed if body.seed is not None else int(time.time() * 1000) % 100000
        try:
            game = SnakeGame(width=body.width, height=body.height, seed=seed)
            policy = LayaPolicy(agent, guarded=body.guarded, prompt=body.prompt)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        sid = uuid.uuid4().hex[:12]
        with sessions_lock:
            while len(sessions) >= MAX_SNAKE_SESSIONS:
                oldest = min(sessions, key=lambda s: sessions[s]["created"])
                del sessions[oldest]
            sessions[sid] = {
                "game": game,
                "policy": policy,
                "created": time.time(),
                "stats": {"moves": 0, "interventions": 0, "inference_ms_total": 0.0},
            }
        return {"session": sid, "seed": seed, "state": game.snapshot()}

    @app.post("/api/snake/step")
    def snake_step(body: SnakeStepBody):
        with sessions_lock:
            entry = sessions.get(body.session)
        if entry is None:
            raise HTTPException(404, "Unknown or expired snake session")
        game, policy, stats = entry["game"], entry["policy"], entry["stats"]
        if not game.alive or game.won:
            return {"state": game.snapshot(), "stats": stats, "done": True}
        try:
            decision = policy.decide(game)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(500, str(exc))
        game.step(decision.executed)
        stats["moves"] += 1
        stats["interventions"] += int(decision.intervened)
        stats["inference_ms_total"] += decision.inference_ms
        return {
            "decision": decision.to_dict(),
            "state": game.snapshot(),
            "stats": stats,
            "done": not game.alive or game.won,
        }

    flappy_sessions: Dict[str, Dict[str, Any]] = {}

    @app.post("/api/flappy/new")
    def flappy_new(body: FlappyNewBody):
        agent = registry.get(body.model)
        seed = body.seed if body.seed is not None else int(time.time() * 1000) % 100000
        game = FlappyGame(seed=seed)
        policy = LayaFlappyPolicy(agent, guarded=body.guarded)
        sid = uuid.uuid4().hex[:12]
        with sessions_lock:
            while len(flappy_sessions) >= MAX_SNAKE_SESSIONS:
                oldest = min(flappy_sessions, key=lambda s: flappy_sessions[s]["created"])
                del flappy_sessions[oldest]
            flappy_sessions[sid] = {
                "game": game,
                "policy": policy,
                "created": time.time(),
                "stats": {"steps": 0, "flaps": 0, "interventions": 0, "inference_ms_total": 0.0},
            }
        return {"session": sid, "seed": seed, "state": game.snapshot()}

    @app.post("/api/flappy/step")
    def flappy_step(body: SnakeStepBody):
        with sessions_lock:
            entry = flappy_sessions.get(body.session)
        if entry is None:
            raise HTTPException(404, "Unknown or expired flappy session")
        game, policy, stats = entry["game"], entry["policy"], entry["stats"]
        if not game.alive:
            return {"state": game.snapshot(), "stats": stats, "done": True}
        try:
            decision = policy.decide(game)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(500, str(exc))
        game.step(decision.executed == "up")
        stats["steps"] += 1
        stats["flaps"] += int(decision.executed == "up")
        stats["interventions"] += int(decision.intervened)
        stats["inference_ms_total"] += decision.inference_ms
        return {
            "decision": decision.to_dict(),
            "state": game.snapshot(),
            "stats": stats,
            "done": not game.alive,
        }

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

    return app
