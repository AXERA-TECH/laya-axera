"""FastAPI web demo: a typed-decision playground and Laya-driven games."""

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .agent import Agent
from .bird.game import BirdGame
from .bird.policy import LayaBirdPolicy
from .blocks.game import BlocksGame
from .blocks.policy import LayaBlocksPolicy
from .bricks.game import ACTIONS as BRICK_ACTIONS
from .bricks.game import BricksGame
from .bricks.policy import LayaBricksPolicy
from .paddle.policy import LayaPaddlePolicy
from .snake.game import DIRECTIONS as SNAKE_DIRECTIONS
from .snake.game import SnakeGame
from .snake.policy import LayaPolicy
from .tank.policy import LayaTankPolicy

WEB_DIR = Path(__file__).parent / "web"
MAX_SESSIONS = 16
BIRD_ACTIONS = ("up", "down")


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


class GameNewBody(BaseModel):
    model: Optional[str] = None
    seed: Optional[int] = None
    guarded: bool = True


class GameStepBody(BaseModel):
    session: str
    # Manual mode: the player's action is executed; the model still decides so
    # the page can show what it would have done.
    action: Optional[str] = None


class BlocksPlaceBody(BaseModel):
    session: str
    rotation: int
    col: int


class PaddleDecideBody(BaseModel):
    model: Optional[str] = None
    court: Dict[str, Any]
    ball: Dict[str, Any]
    ai: Dict[str, Any]


class TankDecideBody(BaseModel):
    model: Optional[str] = None
    grid: List[List[int]]
    ai: Dict[str, Any]
    player: Dict[str, Any]
    bullets: List[Dict[str, Any]] = []


class Sessions:
    """A bounded, thread-safe table of live games; the oldest is evicted first."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def add(self, game, policy, stats) -> str:
        sid = uuid.uuid4().hex[:12]
        with self._lock:
            while len(self._items) >= MAX_SESSIONS:
                oldest = min(self._items, key=lambda s: self._items[s]["created"])
                del self._items[oldest]
            self._items[sid] = {
                "game": game,
                "policy": policy,
                "created": time.time(),
                "stats": stats,
            }
        return sid

    def get(self, sid: str):
        with self._lock:
            entry = self._items.get(sid)
        if entry is None:
            raise HTTPException(404, f"Unknown or expired {self.kind} session")
        return entry["game"], entry["policy"], entry["stats"]


def _seed(value: Optional[int]) -> int:
    return value if value is not None else int(time.time() * 1000) % 100000


def _decide(policy, game):
    try:
        return policy.decide(game)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(500, str(exc))


def _chosen(decision, manual: Optional[str], allowed, stats) -> str:
    """The action to execute: the player's in manual mode, otherwise the model's."""
    if manual is None:
        stats["interventions"] += int(decision.intervened)
        return decision.executed
    if manual not in allowed:
        raise HTTPException(400, f"action must be one of {list(allowed)}")
    stats["matches"] = stats.get("matches", 0) + int(manual == decision.executed)
    return manual


def create_app(checkpoints: Dict[str, Path], *, device_id=0, provider=None) -> FastAPI:
    registry = ModelRegistry(checkpoints, device_id, provider)
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

    # ---- snake ----
    snake = Sessions("snake")

    @app.post("/api/snake/new")
    def snake_new(body: SnakeNewBody):
        agent = registry.get(body.model)
        seed = _seed(body.seed)
        try:
            game = SnakeGame(width=body.width, height=body.height, seed=seed)
            policy = LayaPolicy(agent, guarded=body.guarded)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        stats = {"moves": 0, "interventions": 0, "inference_ms_total": 0.0}
        return {"session": snake.add(game, policy, stats), "seed": seed, "state": game.snapshot()}

    @app.post("/api/snake/step")
    def snake_step(body: GameStepBody):
        game, policy, stats = snake.get(body.session)
        if not game.alive or game.won:
            return {"state": game.snapshot(), "stats": stats, "done": True}
        decision = _decide(policy, game)
        action = _chosen(decision, body.action, SNAKE_DIRECTIONS, stats)
        game.step(action)
        stats["moves"] += 1
        stats["inference_ms_total"] += decision.inference_ms
        return {
            "decision": decision.to_dict(),
            "executed": action,
            "manual": body.action is not None,
            "state": game.snapshot(),
            "stats": stats,
            "done": not game.alive or game.won,
        }

    # ---- bird ----
    bird = Sessions("bird")

    @app.post("/api/bird/new")
    def bird_new(body: GameNewBody):
        agent = registry.get(body.model)
        seed = _seed(body.seed)
        game = BirdGame(seed=seed)
        stats = {"steps": 0, "flaps": 0, "interventions": 0, "inference_ms_total": 0.0}
        sid = bird.add(game, LayaBirdPolicy(agent, guarded=body.guarded), stats)
        return {"session": sid, "seed": seed, "state": game.snapshot()}

    @app.post("/api/bird/step")
    def bird_step(body: GameStepBody):
        game, policy, stats = bird.get(body.session)
        if not game.alive:
            return {"state": game.snapshot(), "stats": stats, "done": True}
        decision = _decide(policy, game)
        action = _chosen(decision, body.action, BIRD_ACTIONS, stats)
        game.step(action == "up")
        stats["steps"] += 1
        stats["flaps"] += int(action == "up")
        stats["inference_ms_total"] += decision.inference_ms
        return {
            "decision": decision.to_dict(),
            "executed": action,
            "manual": body.action is not None,
            "trail": game.trail,
            "state": game.snapshot(),
            "stats": stats,
            "done": not game.alive,
        }

    # ---- falling blocks ----
    blocks = Sessions("blocks")

    @app.post("/api/blocks/new")
    def blocks_new(body: GameNewBody):
        agent = registry.get(body.model)
        seed = _seed(body.seed)
        game = BlocksGame(seed=seed)
        stats = {"pieces": 0, "interventions": 0, "inference_ms_total": 0.0}
        sid = blocks.add(game, LayaBlocksPolicy(agent, guarded=body.guarded), stats)
        return {"session": sid, "seed": seed, "state": game.snapshot()}

    @app.post("/api/blocks/step")
    def blocks_step(body: GameStepBody):
        game, policy, stats = blocks.get(body.session)
        if not game.alive:
            return {"state": game.snapshot(), "stats": stats, "done": True}
        decision = _decide(policy, game)
        game.apply(decision.candidates[decision.executed])
        stats["pieces"] += 1
        stats["interventions"] += int(decision.intervened)
        stats["inference_ms_total"] += decision.inference_ms
        return {
            "decision": decision.to_dict(),
            "state": game.snapshot(),
            "stats": stats,
            "done": not game.alive,
        }

    @app.post("/api/blocks/place")
    def blocks_place(body: BlocksPlaceBody):
        """Manual mode: the player places, the model rates the move and shows its own pick."""
        game, policy, stats = blocks.get(body.session)
        if not game.alive:
            return {"state": game.snapshot(), "stats": stats, "done": True}
        user_cand = game.evaluate(body.rotation, body.col)
        if user_cand is None:
            raise HTTPException(400, "Illegal placement for the current piece")
        try:
            rated, inf_a, _ = policy.rate(game.candidates())
            your, inf_b, _ = policy.rate([user_cand])
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(500, str(exc))
        model_pick = max(range(len(rated)), key=lambda i: rated[i]["p_good"])
        match = your[0]["cells"] == rated[model_pick]["cells"]
        game.apply(user_cand)
        stats["pieces"] += 1
        stats["matches"] = stats.get("matches", 0) + int(match)
        stats["inference_ms_total"] += inf_a + inf_b
        return {
            "your": your[0],
            "candidates": rated,
            "model_pick": model_pick,
            "match": match,
            "inference_ms": round(inf_a + inf_b, 3),
            "state": game.snapshot(),
            "stats": stats,
            "done": not game.alive,
        }

    # ---- bricks ----
    bricks = Sessions("bricks")

    @app.post("/api/bricks/new")
    def bricks_new(body: GameNewBody):
        agent = registry.get(body.model)
        seed = _seed(body.seed)
        game = BricksGame(seed=seed)
        stats = {"steps": 0, "interventions": 0, "inference_ms_total": 0.0}
        sid = bricks.add(game, LayaBricksPolicy(agent, guarded=body.guarded), stats)
        return {"session": sid, "seed": seed, "state": game.snapshot()}

    @app.post("/api/bricks/step")
    def bricks_step(body: GameStepBody):
        game, policy, stats = bricks.get(body.session)
        if not game.alive or game.won:
            return {"state": game.snapshot(), "stats": stats, "done": True}
        decision = _decide(policy, game)
        action = _chosen(decision, body.action, BRICK_ACTIONS, stats)
        game.step(action)
        stats["steps"] += 1
        stats["inference_ms_total"] += decision.inference_ms
        return {
            "decision": decision.to_dict(),
            "executed": action,
            "manual": body.action is not None,
            "trail": game.trail,
            "state": game.snapshot(),
            "stats": stats,
            "done": not game.alive or game.won,
        }

    # ---- tank duel: the page runs the arena, the server only picks the AI's move ----
    @app.post("/api/tank/decide")
    def tank_decide(body: TankDecideBody):
        policy = LayaTankPolicy(registry.get(body.model))
        try:
            return policy.decide(body.grid, body.ai, body.player, body.bullets)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    # ---- paddle duel: the page runs the table, the server only moves the AI paddle ----
    @app.post("/api/paddle/decide")
    def paddle_decide(body: PaddleDecideBody):
        policy = LayaPaddlePolicy(registry.get(body.model))
        try:
            return policy.decide(body.court, body.ball, body.ai)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

    return app
