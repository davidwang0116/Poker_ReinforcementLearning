"""
random_agent.py — Uniform-random baseline agent.

Selects uniformly at random from the set of *legal* actions at
every decision point.  This is the weakest meaningful baseline:
any learning agent should eventually outperform it.

────────────────────────────────────────────────────────────
Lecture-note connections
────────────────────────────────────────────────────────────

[LECTURE: ModelFreeRL — A random policy is the simplest
 possible "behavior policy".  In HW2 we used a random policy
 for rollouts before learning a value function.  Here the
 random agent serves as the performance *floor*: every trained
 agent must beat random play to be considered useful.]

[LECTURE: bandits — Even in the simplest setting (multi-armed
 bandits), uniformly random action selection is sub-optimal.
 In a game like poker, a random player will lose badly against
 any opponent that learns even a rudimentary strategy.]
"""

from __future__ import annotations

from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Uniform-random agent — the simplest possible baseline.

    >>> from poker_env import PokerEnv, play_episode
    >>> env = PokerEnv()
    >>> agents = {
    ...     0: RandomAgent(player_id=0, num_actions=env.num_actions),
    ...     1: RandomAgent(player_id=1, num_actions=env.num_actions),
    ... }
    >>> result = play_episode(env, agents)
    >>> print(result.returns)  # roughly zero-sum, noisy around 0
    """

    def step(self, time_step) -> int:
        """Pick a uniformly random legal action.

        [LECTURE: ModelFreeRL — The ε-greedy policy with ε=1.0 is
         equivalent to uniformly random action selection.  This is
         the "fully exploratory" extreme; learning algorithms start
         near ε=1 and anneal toward ε≈0 (greedy) over training.]
        """
        return self._rng.choice(time_step.legal_actions)
