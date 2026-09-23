"""Breakout demo: deterministic physics, landing-point planner, Laya moves the paddle."""

from .game import BreakoutGame
from .policy import BreakoutDecision, LayaBreakoutPolicy

__all__ = ["BreakoutGame", "BreakoutDecision", "LayaBreakoutPolicy"]
