"""Paddle duel: the browser runs the table, Laya moves the AI paddle."""

from .planner import ACTIONS, plan
from .policy import LayaPaddlePolicy

__all__ = ["ACTIONS", "plan", "LayaPaddlePolicy"]
