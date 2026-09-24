"""Brick-breaker demo: deterministic physics, landing-point planner, Laya moves the paddle."""

from .game import BricksGame
from .policy import BricksDecision, LayaBricksPolicy

__all__ = ["BricksGame", "BricksDecision", "LayaBricksPolicy"]
