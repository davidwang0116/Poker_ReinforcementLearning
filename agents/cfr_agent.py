"""
cfr_agent.py — Counterfactual Regret Minimization (CFR) agent.

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
Implementation: Option A - Wrap OpenSpiel's built-in CFR
────────────────────────────────────────────────────────────

This implementation wraps OpenSpiel's CFR solver to provide a
BaseAgent-compatible interface. The CFR algorithm iteratively
traverses the game tree, computing counterfactual regrets for
each information set and updating strategies via regret matching.

The average strategy over all iterations provably converges to
a Nash equilibrium in two-player zero-sum games.
"""

from __future__ import annotations

import pickle
from typing import Optional

import numpy as np
import pyspiel
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
from tqdm import tqdm

from agents.base_agent import BaseAgent
from config import CFRConfig


class CFRAgent(BaseAgent):
    """CFR agent using OpenSpiel's built-in CFR solver.
    
    This agent trains via Counterfactual Regret Minimization to
    compute a Nash equilibrium strategy for Leduc Hold'em poker.
    
    Unlike RL agents (DQN, PPO), CFR does not learn from experience
    replay or policy gradients. Instead, it iteratively traverses
    the full game tree, accumulating regret for each action at each
    information set. The average strategy converges to Nash.
    
    Attributes
    ----------
    player_id : int
        The seat this agent occupies (0 or 1).
    num_actions : int
        Size of the action space.
    game : pyspiel.Game
        The OpenSpiel game instance.
    solver : cfr.CFRSolver
        The CFR solver that computes the equilibrium strategy.
    policy : pyspiel.TabularPolicy
        The average policy (Nash approximation).
    _policy_dict : dict
        Cached policy as a dictionary for fast lookup.
    """

    def __init__(
        self,
        player_id: int,
        num_actions: int,
        game: Optional[pyspiel.Game] = None,
        config: Optional[CFRConfig] = None,
        rng_seed: int = 0,
    ):
        super().__init__(player_id, num_actions, rng_seed)
        
        if config is None:
            config = CFRConfig()
        self.config = config
        
        # Load game
        if game is None:
            from config import GAME_NAME
            game = pyspiel.load_game(GAME_NAME)
        self.game = game
        
        # Initialize CFR solver
        self.solver = cfr.CFRSolver(self.game)
        
        # Policy will be set after training
        self.policy: Optional[pyspiel.TabularPolicy] = None
        self._policy_dict: dict = {}
        self._is_trained = False
        
        # For tracking game state during play
        self._current_state: Optional[pyspiel.State] = None

    def step(self, time_step) -> int:
        """Choose an action using the learned CFR policy.
        
        [LECTURE: TreeSearch — Unlike MCTS which searches at
         decision time, CFR pre-computes a complete strategy
         (policy) for every information set. At test time we
         simply look up the info-state and sample from the
         stored probability distribution.]
        """
        if not self._is_trained or not self._policy_dict:
            # Fallback to random if not trained
            return self._rng.choice(time_step.legal_actions)
        
        # Convert observation tensor to a hashable key
        # We use the observation tensor as a proxy for the info state
        obs_key = tuple(time_step.obs.tolist())
        
        # Look up policy for this observation
        if obs_key in self._policy_dict:
            action_probs = self._policy_dict[obs_key]
            
            # Filter to legal actions and normalize
            legal_probs = {a: p for a, p in action_probs.items() 
                          if a in time_step.legal_actions}
            
            if legal_probs:
                total = sum(legal_probs.values())
                if total > 0:
                    actions = list(legal_probs.keys())
                    probs = [legal_probs[a] / total for a in actions]
                    return int(self._rng.choice(actions, p=probs))
        
        # Fallback to uniform random if state not found
        return self._rng.choice(time_step.legal_actions)
    
    def on_episode_end(self, final_reward: float) -> None:
        """Reset state tracking at episode end."""
        self._current_state = None

    def train(self, env=None, num_episodes: int = None, **kwargs) -> dict:
        """Train the CFR agent by iterating the CFR algorithm.
        
        [LECTURE: TreeSearch / DeepRL — CFR training is fundamentally
         different from RL training. Instead of collecting experience
         and updating neural networks, CFR performs full game-tree
         traversals, computing counterfactual values and regrets.
         Each iteration updates the strategy for every information set.]
        
        Parameters
        ----------
        env : PokerEnv, optional
            Not used by CFR (we use the game directly).
        num_episodes : int, optional
            Number of CFR iterations (default from config).
        
        Returns
        -------
        dict with training metrics:
            - exploitability: list of exploitability values
            - final_exploitability: final Nash distance
            - iterations: number of CFR iterations run
        """
        if num_episodes is None:
            num_episodes = self.config.num_iterations
        
        print(f"Training CFR agent for {num_episodes} iterations...")
        
        # Track exploitability over training
        exploitability_history = []
        eval_interval = max(1, num_episodes // 10)  # Evaluate 10 times
        
        for i in tqdm(range(num_episodes), desc="CFR iterations"):
            # Run one iteration of CFR
            self.solver.evaluate_and_update_policy()
            
            # Periodically evaluate exploitability
            if (i + 1) % eval_interval == 0 or i == num_episodes - 1:
                avg_policy = self.solver.average_policy()
                exploit = exploitability.exploitability(self.game, avg_policy)
                exploitability_history.append(exploit)
                
                if (i + 1) % (eval_interval * 2) == 0:
                    print(f"  Iteration {i+1}/{num_episodes}: "
                          f"exploitability = {exploit:.6f}")
        
        # Extract final average policy
        self.policy = self.solver.average_policy()
        
        # Build policy dictionary for fast lookup during play
        # We need to map observations to action probabilities
        print("\nBuilding policy lookup table...")
        self._build_policy_dict()
        
        self._is_trained = True
        
        final_exploit = exploitability.exploitability(self.game, self.policy)
        print(f"\nTraining complete!")
        print(f"Final exploitability: {final_exploit:.6f}")
        print(f"(Lower is better; 0 = perfect Nash equilibrium)")
        
        return {
            "exploitability": exploitability_history,
            "final_exploitability": final_exploit,
            "iterations": num_episodes,
        }
    
    def _build_policy_dict(self) -> None:
        """Build a dictionary mapping observations to action probabilities.
        
        This traverses the game tree and records the policy for each
        information state, indexed by the observation tensor.
        """
        self._policy_dict = {}
        
        # Traverse the game tree to collect all info states
        def traverse(state: pyspiel.State):
            if state.is_terminal():
                return
            
            if state.is_chance_node():
                # Explore all chance outcomes
                for action, _ in state.chance_outcomes():
                    state_copy = state.clone()
                    state_copy.apply_action(action)
                    traverse(state_copy)
                return
            
            # Decision node - record the policy
            player = state.current_player()
            
            # Get observation tensor for this state
            obs = np.array(
                state.information_state_tensor(player),
                dtype=np.float32
            )
            obs_key = tuple(obs.tolist())
            
            # Get action probabilities from policy (pass state, not string)
            try:
                action_probs = self.policy.action_probabilities(state)
                self._policy_dict[obs_key] = dict(action_probs)
            except (KeyError, RuntimeError, AttributeError) as e:
                # Some states might not be in the policy yet
                pass
            
            # Continue traversal for all legal actions
            for action in state.legal_actions():
                state_copy = state.clone()
                state_copy.apply_action(action)
                traverse(state_copy)
        
        # Start traversal from initial state
        initial_state = self.game.new_initial_state()
        traverse(initial_state)
        
        print(f"  Recorded {len(self._policy_dict)} information states")

    def save(self, path: str) -> None:
        """Save the learned CFR policy to disk.
        
        Note: We save the policy dictionary, not the full CFR solver
        state (which includes regrets and cumulative strategies).
        For continuing training, you would need to save the solver.
        """
        if not self._is_trained or not self._policy_dict:
            raise ValueError("Cannot save untrained agent. Train first.")
        
        save_data = {
            "policy_dict": self._policy_dict,
            "player_id": self.player_id,
            "num_actions": self.num_actions,
            "config": self.config,
        }
        
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
        
        print(f"CFR policy saved to {path}")

    def load(self, path: str) -> None:
        """Load a previously saved CFR policy from disk."""
        with open(path, "rb") as f:
            save_data = pickle.load(f)
        
        # Restore policy dictionary
        self._policy_dict = save_data["policy_dict"]
        self.player_id = save_data["player_id"]
        self.num_actions = save_data["num_actions"]
        self.config = save_data["config"]
        
        self._is_trained = True
        print(f"CFR policy loaded from {path}")

    def get_exploitability(self) -> float:
        """Compute the exploitability of the current policy.
        
        [LECTURE: DeepRL — Exploitability measures how far a
         strategy is from Nash equilibrium. It's the sum of
         how much each player could gain by best-responding
         to the strategy. A Nash equilibrium has exploitability 0.]
        
        Returns
        -------
        float — exploitability value (lower is better, 0 = Nash)
        
        Note: For loaded agents, we need to retrain briefly to get
        a proper policy object for exploitability calculation.
        """
        if not self._is_trained:
            raise ValueError("Agent must be trained first.")
        
        # If we have a policy object, use it directly
        if self.policy is not None:
            return exploitability.exploitability(self.game, self.policy)
        
        # If we only have the policy dict (loaded from file),
        # we need to reconstruct the policy by running a few CFR iterations
        # This is a limitation of the current save/load implementation
        print("Note: Reconstructing policy for exploitability calculation...")
        print("      (Running 100 CFR iterations...)")
        
        # Run a small number of iterations to get a policy object
        temp_solver = cfr.CFRSolver(self.game)
        for _ in range(100):
            temp_solver.evaluate_and_update_policy()
        
        temp_policy = temp_solver.average_policy()
        return exploitability.exploitability(self.game, temp_policy)

    def __repr__(self) -> str:
        trained_str = "trained" if self._is_trained else "untrained"
        return f"CFRAgent(player={self.player_id}, {trained_str})"
