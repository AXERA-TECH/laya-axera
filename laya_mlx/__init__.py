"""Laya typed decisions on Apple silicon with MLX."""

from .agent import Agent, RLAgent, load
from .email import clean_email_body, email_state
from .lang import analyse as detect_language
from .lang import detect_script, is_english
from .presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from .router import DEFAULT_MODELS, RouteDecision, Router
from .shortlist import embed_fn_from_agent, predict_shortlist, shortlist_choice

__version__ = "0.2.0"
__all__ = [
    "Agent",
    "RLAgent",
    "load",
    "Router",
    "RouteDecision",
    "DEFAULT_MODELS",
    "shortlist_choice",
    "predict_shortlist",
    "embed_fn_from_agent",
    "detect_language",
    "detect_script",
    "is_english",
    "clean_email_body",
    "email_state",
    "email_questions",
    "guard_questions",
    "moderation_questions",
    "router_questions",
    "triage_questions",
]
