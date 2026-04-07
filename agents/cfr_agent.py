"""
cfr_agent.py — Counterfactual Regret Minimization (CFR) agent.

*** PERSON 3: Implement this file. ***

────────────────────────────────────────────────────────────
Lecture-note connections
────────────────────────────────────────────────────────────

[LECTURE: DeepRL — "Extension to 2 player games":
   V(s) = R(s) + γ max_a Σ P(s'|s,a)(R(s') + γ min_{a'} Σ P(s''|s',a') V(s''))
 In perfect-information games this is solved by minimax.
 In *imperfect*-information games (like poker), minimax
 doesn't directly apply because players can't see the full
 state.  CFR solves this by working over *information sets*
 rather than individual states.]

[LECTURE: TreeSearch — Tree search (minimax, alpha-beta, MCTS)
 explores the game tree.  CFR is conceptually similar: it
 traverses the game tree, but instead of backing up values
 it accumulates *regret* for each action at each information set.
 The strategy that minimises cumulative regret converges
 to a Nash equilibrium (Zinkevich et al., 2007).]

────────────────────────────────────────────────────────────
Implementation guide
────────────────────────────────────────────────────────────

**Option A: Wrap OpenSpiel's built-in CFR**
   OpenSpiel provides `open_spiel.python.algorithms.cfr` which
   implements vanilla CFR, CFR+, and linear CFR.

   Steps:
   1. Create a CFR solver:
        from open_spiel.python.algorithms import cfr
        solver = cfr.CFRSolver(game)
   2. Run iterations:
        for i in range(num_iterations):
            solver.evaluate_and_update_policy()
   3. Extract average policy:
        avg_policy = solver.average_policy()
   4. Wrap in a BaseAgent that looks up the info-state string
      and samples from the policy.
   5. Use `evaluate_exploitability(avg_policy, game)` from
      evaluate.py to measure convergence to Nash.

**Option B: Implement CFR from scratch**
   For deeper understanding, implement the CFR algorithm:
   - Traverse game tree recursively.
   - At each information set, maintain cumulative regret
     and cumulative strategy.
   - Update strategy via regret matching:
       σ(a) = max(R(a), 0) / Σ_b max(R(b), 0)
   - Average strategy converges to Nash.

**Option C: NFSP (Neural Fictitious Self-Play)**
   Combines RL (DQN for best-response) with supervised learning
   (to approximate the average strategy).  More scalable than
   tabular CFR but harder to implement.
   OpenSpiel provides `open_spiel.python.algorithms.nfsp`.
"""

from __future__ import annotations

# from open_spiel.python.algorithms import cfr
# from open_spiel.python.algorithms import exploitability
# import pyspiel
# import numpy as np
# from agents.base_agent import BaseAgent
# from config import CFRConfig


# class CFRAgent(BaseAgent):
#     """TODO — Person 3"""
#     pass
