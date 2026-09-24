"""Tank duel: the browser runs the arena, Laya picks the AI tank's next action."""

from .planner import ORDER, plan
from .policy import LayaTankPolicy

__all__ = ["ORDER", "plan", "LayaTankPolicy"]
