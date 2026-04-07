"""
config.py — Project-wide configuration and hyper-parameters.

Every teammate imports from here so experiments are reproducible.
Modify the values below to tune experiments; do NOT hard-code
hyper-parameters elsewhere.
"""

from dataclasses import dataclass, field
from typing import List

# ──────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────

# [LECTURE: MDPs_1 — the MDP is defined by ⟨S, A, T, R, γ⟩.
#  OpenSpiel games generalise this to imperfect-information
#  extensive-form games with *chance nodes* (card deals) and
#  *information sets* (what a player can observe).  Leduc
#  Hold'em is small enough for tabular CFR yet rich enough
#  to require function approximation for RL methods.]

GAME_NAME: str = "leduc_poker"
"""OpenSpiel game identifier.  Alternatives to try:
    - "kuhn_poker"        (even simpler, 1 betting round, 3-card deck)
    - "leduc_poker"       (2 rounds, 6-card deck — our default)
    - "universal_poker"   (configurable; much larger)
"""

NUM_PLAYERS: int = 2
"""Number of players.  Leduc poker is a 2-player game."""

# ──────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────

SEED: int = 42
"""Global random seed for reproducibility."""

NUM_TRAIN_EPISODES: int = 100_000
"""Total training episodes for learning agents (DQN / PPO)."""

EVAL_EVERY: int = 5_000
"""Evaluate agents every this many episodes during training."""

# ──────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────

NUM_EVAL_EPISODES: int = 10_000
"""Number of episodes per evaluation run (for mean reward / win-rate)."""

# ──────────────────────────────────────────────────────────
# DQN hyper-parameters  (Person 2)
# ──────────────────────────────────────────────────────────
@dataclass
class DQNConfig:
    """
    [LECTURE: DeepRL — DQN uses experience replay and a target
     network to stabilise training of Q(s, a; θ).  The target
     network is a *delayed copy* of the online network, updated
     every `target_update_freq` steps.  This breaks the harmful
     correlation between the current Q-estimate and the TD target,
     which otherwise causes divergence (the "deadly triad").]
    """
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    lr: float = 1e-3
    gamma: float = 1.0  # poker is episodic; no need to discount
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 80_000
    epsilon_decay = 0.99995
    replay_buffer_size: int = 50_000
    batch_size: int = 128
    target_update_freq: int = 1_000  # steps between target-net syncs


# ──────────────────────────────────────────────────────────
# PPO hyper-parameters  (Person 2)
# ──────────────────────────────────────────────────────────
@dataclass
class PPOConfig:
    """
    [LECTURE: PolicySearch / PGinPractice — PPO clips the
     likelihood-ratio objective to prevent destructively large
     policy updates (cf. PGinPractice.pdf, TRPO → PPO).
     The advantage function Â = Q - V reduces variance
     compared to raw returns (cf. PolicySearch.pdf, eq. 11.44).]
    """
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    lr: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    epochs_per_update: int = 4
    batch_size: int = 256
    rollout_length: int = 512


# ──────────────────────────────────────────────────────────
# CFR hyper-parameters  (Person 3)
# ──────────────────────────────────────────────────────────
@dataclass
class CFRConfig:
    """
    [LECTURE: TreeSearch / DeepRL — Counterfactual Regret
     Minimisation (CFR) iterates over the game tree, accumulating
     *regret* for each information set.  The average strategy
     converges to a Nash equilibrium in two-player zero-sum games.
     This connects to the minimax formulation on DeepRL.pdf
     slide "Extension to 2 player games":
       V(s) = R(s) + γ max_a Σ P(s'|s,a)(R(s') + γ min_{a'} Σ P(s''|s',a') V(s''))
     CFR generalises this to *stochastic* imperfect-info games.]
    """
    num_iterations: int = 100_000
    # For NFSP (Neural Fictitious Self-Play) if used instead:
    nfsp_hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    nfsp_lr: float = 1e-3
    anticipatory_param: float = 0.1
