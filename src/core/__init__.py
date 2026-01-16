"""
Core module - contains core components of the debate system
"""

from .agent import Agent
from .debate import Debate
from .players import DebatePlayer

__all__ = ["Agent", "Debate", "DebatePlayer"]
