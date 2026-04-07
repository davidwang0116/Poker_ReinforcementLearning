"""
agents — Plug-in agent implementations.

Every concrete agent subclasses `BaseAgent` and implements at minimum:
    step(ts: TimeStep) → int          choose an action
    on_episode_end(final_reward)       end-of-episode bookkeeping

Learning agents additionally implement:
    train()          run / continue training
    save(path)       serialise learned parameters
    load(path)       restore learned parameters
"""

from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent

__all__ = ["BaseAgent", "RandomAgent"]
