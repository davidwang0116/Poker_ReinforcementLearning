"""
cfr_agent.py — Counterfactual Regret Minimization (CFR) agent.

Wraps OpenSpiel's built-in CFR solver to find Nash equilibrium
strategies for Leduc Hold'em poker.

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
"""

from __future__ import annotations

import pickle
from typing import Optional
import numpy as np
import pyspiel
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability

from agents.base_agent import BaseAgent
from config import CFRConfig


class CFRAgent(BaseAgent):
    """
    CFR agent that computes Nash equilibrium strategies.
    
    Unlike DQN/PPO which learn through environment interaction,
    CFR computes the optimal strategy by iteratively traversing
    the full game tree and minimizing regret at each information set.
    
    [LECTURE: DeepRL — In two-player zero-sum games, the Nash
     equilibrium is the minimax solution.  CFR provably converges
     to Nash by minimising *counterfactual regret*: the regret
     for not having played a different action at each information
     set, weighted by the probability of reaching that set.]
    """
    
    def __init__(
        self,
        player_id: int,
        num_actions: int,
        obs_size: int,
        config: Optional[CFRConfig] = None,
        game: Optional[pyspiel.Game] = None,
    ):
        """
        Initialize CFR agent.
        
        Parameters
        ----------
        player_id : int
            Which player this agent controls (0 or 1)
        num_actions : int
            Size of the action space
        obs_size : int
            Size of observation tensor (unused for CFR but kept for interface compatibility)
        config : CFRConfig, optional
            Hyperparameters for CFR
        game : pyspiel.Game, optional
            The game instance (required for CFR training)
        """
        super().__init__(player_id, num_actions)
        
        self.config = config if config is not None else CFRConfig()
        self.game = game
        self.obs_size = obs_size
        
        # CFR solver (created when training starts)
        self.solver: Optional[cfr.CFRSolver] = None
        
        # Average policy (extracted after training)
        self.policy: Optional[pyspiel.TabularPolicy] = None
        
        # Track training progress
        self.num_iterations_trained = 0
        
    def step(self, state) -> int:
        """
        Choose an action using the CFR average policy.
        
        Parameters
        ----------
        state : pyspiel.State or TimeStep
            Current game state
            
        Returns
        -------
        int — action index sampled from the policy
        
        [LECTURE: ModelFreeRL — Unlike ε-greedy (exploration) or
         greedy (exploitation), CFR uses a *mixed strategy*: a
         probability distribution over actions.  This is closer to
         the softmax policies in PolicySearch.pdf than the
         deterministic policies in Q-learning.]
        """
        if self.policy is None:
            raise RuntimeError(
                "CFRAgent has not been trained yet. Call train() first."
            )
        
        # Handle both pyspiel.State and TimeStep inputs
        if hasattr(state, 'information_state_string'):
            # Direct OpenSpiel state
            info_state_str = state.information_state_string(self.player_id)
            legal_actions = state.legal_actions(self.player_id)
        else:
            # TimeStep from poker_env
            # We need the actual OpenSpiel state to get info_state_string
            # This is a limitation - for evaluation we'll need the raw state
            raise NotImplementedError(
                "CFRAgent.step() requires a pyspiel.State object. "
                "When using with poker_env, access state via env.get_state()."
            )
        
        # Get action probabilities from the policy
        action_probs = self.policy.action_probabilities(
            info_state_str
        )
        
        # Sample action according to the policy distribution
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        
        # Ensure we only sample legal actions
        legal_action_probs = {a: p for a, p in action_probs.items() if a in legal_actions}
        if not legal_action_probs:
            # Fallback: uniform over legal actions
            return self._rng.choice(legal_actions)
        
        # Renormalize
        total = sum(legal_action_probs.values())
        actions = list(legal_action_probs.keys())
        probs = [p / total for p in legal_action_probs.values()]
        
        return int(self._rng.choice(actions, p=probs))
    
    def train(self, num_iterations: Optional[int] = None) -> dict:
        """
        Run CFR iterations to compute Nash equilibrium.
        
        Parameters
        ----------
        num_iterations : int, optional
            Number of CFR iterations to run.
            Defaults to config.num_iterations.
            
        Returns
        -------
        dict — training metrics including exploitability
        
        [LECTURE: TreeSearch / DeepRL — CFR iteratively traverses
         the game tree.  Each iteration:
         1. Computes counterfactual values for each player
         2. Updates regrets at each information set
         3. Updates the current strategy via regret matching
         The *average* strategy converges to Nash equilibrium.]
        """
        if self.game is None:
            raise ValueError(
                "CFRAgent requires a game instance for training. "
                "Pass it to __init__ via the 'game' parameter."
            )
        
        if num_iterations is None:
            num_iterations = self.config.num_iterations
        
        # Initialize solver if not already done
        if self.solver is None:
            self.solver = cfr.CFRSolver(self.game)
            print(f"Initialized CFR solver for {self.game.get_type().short_name}")
        
        # Run CFR iterations
        print(f"Running {num_iterations} CFR iterations...")
        for i in range(num_iterations):
            self.solver.evaluate_and_update_policy()
            self.num_iterations_trained += 1
            
            # Log progress periodically
            if (i + 1) % (num_iterations // 10) == 0 or i == 0:
                # Compute exploitability to measure convergence
                current_policy = self.solver.average_policy()
                exploit = exploitability.exploitability(self.game, current_policy)
                print(
                    f"  Iteration {self.num_iterations_trained}: "
                    f"exploitability = {exploit:.6f}"
                )
        
        # Extract the average policy (this is the Nash equilibrium approximation)
        self.policy = self.solver.average_policy()
        
        # Compute final exploitability
        final_exploit = exploitability.exploitability(self.game, self.policy)
        
        metrics = {
            'num_iterations': self.num_iterations_trained,
            'exploitability': final_exploit,
        }
        
        print(f"\nCFR training complete!")
        print(f"  Total iterations: {self.num_iterations_trained}")
        print(f"  Final exploitability: {final_exploit:.6f}")
        print(f"  (Lower is better; 0 = perfect Nash equilibrium)")
        
        return metrics
    
    def save(self, path: str) -> None:
        """
        Save the trained CFR policy to disk.
        
        Parameters
        ----------
        path : str
            File path to save to (will add .pkl extension)
        """
        if self.policy is None:
            raise RuntimeError("No policy to save. Train the agent first.")
        
        # Save the policy as a pickle file
        save_path = path if path.endswith('.pkl') else f"{path}.pkl"
        
        # Convert policy to a serializable format
        # Store the policy state dict (information set -> action distribution)
        policy_dict = {}
        for info_state in self.policy.states():
            policy_dict[info_state] = self.policy.action_probabilities(info_state)
        
        save_data = {
            'policy_dict': policy_dict,
            'num_iterations': self.num_iterations_trained,
            'player_id': self.player_id,
            'num_actions': self.num_actions,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"CFR policy saved to {save_path}")
    
    def load(self, path: str) -> None:
        """
        Load a trained CFR policy from disk.
        
        Parameters
        ----------
        path : str
            File path to load from
        """
        load_path = path if path.endswith('.pkl') else f"{path}.pkl"
        
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.num_iterations_trained = save_data['num_iterations']
        self.player_id = save_data['player_id']
        self.num_actions = save_data['num_actions']
        
        # Reconstruct the policy
        # Note: This is a simplified version. For full functionality,
        # we'd need to recreate the TabularPolicy object properly.
        self._policy_dict = save_data['policy_dict']
        
        # Create a mock policy object that can be queried
        class LoadedPolicy:
            def __init__(self, policy_dict):
                self._policy_dict = policy_dict
            
            def action_probabilities(self, info_state):
                return self._policy_dict.get(info_state, {})
            
            def states(self):
                return self._policy_dict.keys()
        
        self.policy = LoadedPolicy(self._policy_dict)
        
        print(f"CFR policy loaded from {load_path}")
        print(f"  Trained for {self.num_iterations_trained} iterations")
    
    def get_exploitability(self) -> float:
        """
        Compute exploitability of the current policy.
        
        Returns
        -------
        float — exploitability value (0 = perfect Nash equilibrium)
        
        [LECTURE: DeepRL — Exploitability measures how much an
         optimal adversary can exploit this policy.  It's defined as:
           exploit(σ) = (BR_0(σ_1) - v_0(σ)) + (BR_1(σ_0) - v_1(σ))
         where BR_i is player i's best response value and v_i is
         their value under the current strategy profile σ.
         For a Nash equilibrium, exploit(σ) = 0.]
        """
        if self.policy is None:
            raise RuntimeError("No policy available. Train the agent first.")
        
        if self.game is None:
            raise ValueError("Game instance required to compute exploitability.")
        
        return exploitability.exploitability(self.game, self.policy)
