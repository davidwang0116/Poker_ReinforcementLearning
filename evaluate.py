"""
evaluate.py — Evaluation harness for comparing poker agents.

Provides three evaluation modes:

  1. **Head-to-head**: Agent A vs Agent B over many episodes.
     Reports mean return, win/draw/loss rates, and confidence
     intervals.

  2. **Round-robin**: Every agent plays every other agent.
     Produces a payoff matrix suitable for ranking.

  3. **Exploitability** (game-theoretic): Measures how far an
     agent's *average strategy* is from a Nash equilibrium.
     Only feasible for small games (Kuhn, Leduc).

────────────────────────────────────────────────────────────
Lecture-note connections
────────────────────────────────────────────────────────────

[LECTURE: DeepRL — "Extension to 2 player games": in a
 zero-sum game the *value* of a strategy is measured by
 how well it does against the *best response*.  The gap
 between a strategy's value and the Nash-equilibrium value
 is called **exploitability**.  Lower exploitability ⟹ closer
 to Nash ⟹ harder to beat by *any* opponent.]

[LECTURE: ModelFreeRL — Evaluation via Monte-Carlo rollouts
 is exactly what we did in HW2 Section 2 (mc_policy_evaluation)
 but applied to two-player games.  We average returns over
 many episodes to estimate E[return | policy].]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from poker_env import PokerEnv, play_episode
from agents.base_agent import BaseAgent
from config import NUM_EVAL_EPISODES, SEED


# ── Data classes for results ────────────────────────────

@dataclass
class HeadToHeadResult:
    """Statistics from one head-to-head evaluation."""
    agent_a_name: str
    agent_b_name: str
    num_episodes: int
    mean_return_a: float
    mean_return_b: float
    std_return_a: float
    std_return_b: float
    win_rate_a: float      # fraction of episodes where return_a > 0
    win_rate_b: float
    draw_rate: float
    elapsed_seconds: float

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f" {self.agent_a_name}  vs  {self.agent_b_name}",
            f" Episodes: {self.num_episodes}",
            f"{'─'*60}",
            f" {self.agent_a_name:>20s}:  mean={self.mean_return_a:+.4f}  "
            f"std={self.std_return_a:.4f}  win={self.win_rate_a:.2%}",
            f" {self.agent_b_name:>20s}:  mean={self.mean_return_b:+.4f}  "
            f"std={self.std_return_b:.4f}  win={self.win_rate_b:.2%}",
            f" {'draws':>20s}:  {self.draw_rate:.2%}",
            f" Time: {self.elapsed_seconds:.1f}s",
            f"{'='*60}",
        ]
        return "\n".join(lines)


@dataclass
class RoundRobinResult:
    """Payoff matrix from a full round-robin tournament."""
    agent_names: List[str]
    payoff_matrix: np.ndarray  # shape (n_agents, n_agents); [i,j] = mean return of i vs j
    details: List[HeadToHeadResult] = field(default_factory=list)


# ── Head-to-head evaluation ─────────────────────────────

def evaluate_head_to_head(
    agent_a: BaseAgent,
    agent_b: BaseAgent,
    num_episodes: int = NUM_EVAL_EPISODES,
    env: Optional[PokerEnv] = None,
    agent_a_name: str = "Agent_A",
    agent_b_name: str = "Agent_B",
) -> HeadToHeadResult:
    """Run *num_episodes* games between agent_a (seat 0) and
    agent_b (seat 1).  Returns aggregated statistics.

    [LECTURE: ModelFreeRL — Monte-Carlo evaluation averages
     returns over N episodes to estimate the policy value.
     The more episodes, the tighter the confidence interval.
     This is the same principle as mc_policy_evaluation in HW2,
     extended to a two-player setting.]
    """
    if env is None:
        env = PokerEnv(seed=SEED + 1000)

    agents = {0: agent_a, 1: agent_b}
    returns_a: List[float] = []
    returns_b: List[float] = []

    t0 = time.time()
    for _ in range(num_episodes):
        result = play_episode(env, agents)
        returns_a.append(result.returns[0])
        returns_b.append(result.returns[1])
    elapsed = time.time() - t0

    ra = np.array(returns_a)
    rb = np.array(returns_b)

    return HeadToHeadResult(
        agent_a_name=agent_a_name,
        agent_b_name=agent_b_name,
        num_episodes=num_episodes,
        mean_return_a=float(ra.mean()),
        mean_return_b=float(rb.mean()),
        std_return_a=float(ra.std()),
        std_return_b=float(rb.std()),
        win_rate_a=float((ra > 0).mean()),
        win_rate_b=float((rb > 0).mean()),
        draw_rate=float((ra == 0).mean()),
        elapsed_seconds=elapsed,
    )


# ── Round-robin tournament ──────────────────────────────

def evaluate_round_robin(
    agents: Dict[str, BaseAgent],
    num_episodes: int = NUM_EVAL_EPISODES,
) -> RoundRobinResult:
    """Every agent plays every other agent.  Builds a payoff matrix.

    Parameters
    ----------
    agents : dict[str, BaseAgent]
        name → agent.  All agents must accept being assigned to
        any player_id (0 or 1).

    [LECTURE: This mirrors the game-theoretic notion of
     evaluating strategy profiles in a normal-form game.
     The payoff matrix shows E[reward] for row-player
     against each column-player.]
    """
    names = sorted(agents.keys())
    n = len(names)
    matrix = np.zeros((n, n), dtype=np.float64)
    details: List[HeadToHeadResult] = []

    env = PokerEnv(seed=SEED + 2000)

    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if i == j:
                continue  # skip self-play for ranking purposes
            # Temporarily override player_ids
            a = agents[name_a]
            b = agents[name_b]
            old_pid_a, old_pid_b = a.player_id, b.player_id
            a.player_id, b.player_id = 0, 1

            h2h = evaluate_head_to_head(
                a, b,
                num_episodes=num_episodes,
                env=env,
                agent_a_name=name_a,
                agent_b_name=name_b,
            )
            matrix[i, j] = h2h.mean_return_a
            details.append(h2h)

            # Restore original player_ids
            a.player_id, b.player_id = old_pid_a, old_pid_b

    return RoundRobinResult(
        agent_names=names,
        payoff_matrix=matrix,
        details=details,
    )


# ── Exploitability (game-theoretic quality) ─────────────

def evaluate_exploitability(
    policy,
    game=None,
) -> float:
    """Compute the exploitability of a *tabular* policy.

    Exploitability measures how much an optimal adversary can
    gain against the given policy.  A Nash-equilibrium policy
    has exploitability 0 in a two-player zero-sum game.

    Parameters
    ----------
    policy : pyspiel.TabularPolicy or compatible
        Must map every information state to a probability
        distribution over actions.
    game : pyspiel.Game, optional
        Defaults to the game defined in config.py.

    Returns
    -------
    float — exploitability (lower is better; 0 = Nash).

    [LECTURE: DeepRL — In two-player zero-sum games the
     minimax solution coincides with the Nash equilibrium.
     Exploitability quantifies the distance from Nash:
       exploit(σ) = max_{σ'_0} v_0(σ'_0, σ_1) + max_{σ'_1} v_1(σ_0, σ'_1)
     where σ_i is player i's strategy and v_i is player i's
     expected payoff.  CFR provably converges to 0 exploitability
     in these games.]
    """
    import pyspiel
    from open_spiel.python.algorithms import exploitability as exploit_lib

    if game is None:
        game = pyspiel.load_game(SEED)
        # Fallback: create default game from config
        from config import GAME_NAME
        game = pyspiel.load_game(GAME_NAME)

    return exploit_lib.exploitability(game, policy)


# ── Pretty-print helpers ────────────────────────────────

def print_round_robin(rr: RoundRobinResult) -> None:
    """Print a formatted payoff matrix to stdout."""
    n = len(rr.agent_names)
    header = "          " + "  ".join(f"{name:>10s}" for name in rr.agent_names)
    print(header)
    print("─" * len(header))
    for i, name in enumerate(rr.agent_names):
        row_vals = "  ".join(
            f"{rr.payoff_matrix[i, j]:>+10.4f}" if i != j else f"{'---':>10s}"
            for j in range(n)
        )
        print(f"{name:>10s}  {row_vals}")
    print()
    # Overall ranking by mean payoff across opponents
    mean_payoffs = []
    for i in range(n):
        vals = [rr.payoff_matrix[i, j] for j in range(n) if i != j]
        mean_payoffs.append(np.mean(vals))
    ranking = sorted(
        zip(rr.agent_names, mean_payoffs), key=lambda x: -x[1]
    )
    print("Ranking (mean payoff across opponents):")
    for rank, (name, mp) in enumerate(ranking, 1):
        print(f"  {rank}. {name:>12s}  {mp:+.4f}")
