"""bird demo: deterministic physics, a 2-ply safety autopilot, and Laya decisions."""

from .game import BirdGame
from .policy import BirdDecision, LayaBirdPolicy

__all__ = ["BirdGame", "BirdDecision", "LayaBirdPolicy"]
