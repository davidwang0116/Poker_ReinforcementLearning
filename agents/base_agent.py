"""
base_agent.py — Abstract base class for all agents.

Person 2 (DQN / PPO) and Person 3 (CFR / NFSP) should subclass
this and implement the abstract methods.

────────────────────────────────────────────────────────────
Lecture-note connections
────────────────────────────────────────────────────────────

[LECTURE: ModelFreeRL — An RL agent maintains a *policy* π(a|s)
 and (optionally) a *value function* V(s) or Q(s,a).
 The `step()` method implements the policy; `train()` updates it
 from experience, mirroring the "RL loop" from the lectures.]

[LECTURE: ModelFreeRL — ε-greedy exploration is the simplest
 exploration strategy for Q-learning and DQN.  We provide a
 helper here so all agents can use it consistently.
 Recall: with probability ε choose a random legal action,
 otherwise choose argmax_a Q(s, a).]
"""

from __future__ import annotations

import abc
from typing import List, Optional

import numpy as np


class BaseAgent(abc.ABC):
    """Interface every agent must satisfy.

    Attributes
    ----------
    player_id : int
        The seat this agent occupies (0 or 1 in 2-player poker).
    num_actions : int
        Size of the *full* action space (including illegal actions;
        agents must respect the legal-actions mask).
    """

    def __init__(self, player_id: int, num_actions: int, rng_seed: int = 0):
        self.player_id = player_id
        self.num_actions = num_actions
        self._rng = np.random.RandomState(rng_seed)

    # ── required ────────────────────────────────────────

    @abc.abstractmethod
    def step(self, time_step) -> int:
        """Choose an action given the current TimeStep.

        Parameters
        ----------
        time_step : poker_env.TimeStep
            Contains obs, legal_actions, legal_actions_mask, etc.

        Returns
        -------
        int — the chosen action index (must be in time_step.legal_actions).
        """
        ...

    def on_episode_end(self, final_reward: float) -> None:
        """Called at the end of every episode with the agent's return.

        Override this to, e.g., store the final reward in a replay
        buffer or update a running average.
        """
        pass  # default: no-op

    # ── optional (learning agents) ──────────────────────

    def train(self, env, num_episodes: int, **kwargs) -> dict:
        """Run (or continue) training.  Return a dict of metrics."""
        raise NotImplementedError("This agent does not support training.")

    def save(self, path: str) -> None:
        """Persist learned parameters to disk."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load previously saved parameters."""
        raise NotImplementedError

    # ── helpers ─────────────────────────────────────────

    def _epsilon_greedy(
        self,
        q_values: np.ndarray,
        legal_actions: List[int],
        epsilon: float,
    ) -> int:
        """ε-greedy action selection restricted to legal actions.

        [LECTURE: ModelFreeRL — ε-greedy is used in Q-learning and
         DQN.  It balances exploration (random action with prob ε)
         and exploitation (greedy action with prob 1−ε).
         Here we mask out illegal actions before taking the argmax.]
        """
        if self._rng.rand() < epsilon:
            return self._rng.choice(legal_actions)
        # Greedy: pick legal action with highest Q-value
        masked_q = np.full(self.num_actions, -np.inf)
        for a in legal_actions:
            masked_q[a] = q_values[a]
        return int(np.argmax(masked_q))

    def _softmax_sample(
        self,
        logits: np.ndarray,
        legal_actions_mask: np.ndarray,
    ) -> int:
        """Sample an action from a masked softmax distribution.

        [LECTURE: PolicySearch — Softmax (Boltzmann) policy:
            π(a|s,θ) = exp(h(s,a,θ)) / Σ_b exp(h(s,b,θ))
         We mask illegal actions to −∞ before softmax so they
         get probability 0.  This is the standard trick for
         policy-gradient methods in discrete action spaces
         (cf. PolicySearch.pdf, eq. 13.2 in Sutton & Barto).]
        """
        # Set illegal actions to large negative value
        masked_logits = np.where(
            legal_actions_mask > 0, logits, -1e9
        )
        # Numerically stable softmax
        shifted = masked_logits - np.max(masked_logits)
        exp_logits = np.exp(shifted)
        probs = exp_logits / exp_logits.sum()
        return int(self._rng.choice(self.num_actions, p=probs))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(player={self.player_id})"
