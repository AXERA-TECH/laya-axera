"""Flappy Bird demo: deterministic physics, a 2-ply safety autopilot, and Laya decisions."""

from .game import FlappyGame
from .policy import FlappyDecision, LayaFlappyPolicy

__all__ = ["FlappyGame", "FlappyDecision", "LayaFlappyPolicy"]
