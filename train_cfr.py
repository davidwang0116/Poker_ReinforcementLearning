"""
train_cfr.py — Training script for CFR agent.

CFR (Counterfactual Regret Minimization) works differently from
DQN/PPO: instead of learning through episodes, it directly computes
the Nash equilibrium by iterating over the game tree.

Usage:
    python train_cfr.py --config configs/cfr.yaml
    python train_cfr.py --iterations 50000  # Override default
"""

import os
import argparse
import yaml
import numpy as np
import pyspiel

from agents.cfr_agent import CFRAgent
from config import CFRConfig


class ConfigObject:
    """Helper to convert dict to object with dot notation."""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, ConfigObject(v) if isinstance(v, dict) else v)


def train_cfr(cfg):
    """
    Train a CFR agent to find Nash equilibrium.
    
    [LECTURE: DeepRL — Unlike model-free RL which learns from
     environment interaction, CFR is a *game-theoretic* algorithm
     that computes the optimal strategy by minimizing regret over
     the entire game tree.  This is only feasible for small games
     like Kuhn and Leduc poker.]
    """
    print("="*60)
    print("CFR Training for Leduc Hold'em Poker")
    print("="*60)
    
    # Set random seed
    np.random.seed(cfg.seed)
    
    # Load game
    game = pyspiel.load_game(cfg.game_name)
    num_actions = game.num_distinct_actions()
    obs_size = game.information_state_tensor_size()
    
    print(f"\nGame: {game.get_type().short_name}")
    print(f"Players: {game.num_players()}")
    print(f"Action space: {num_actions}")
    print(f"Info state size: {obs_size}")
    print(f"Game tree size: ~{game.num_distinct_actions() ** 4} nodes (approximate)")
    
    # Create CFR config
    cfr_config = CFRConfig(
        num_iterations=cfg.cfr.num_iterations
    )
    
    # Create CFR agent
    # Note: player_id doesn't matter for CFR since it computes strategies for both players
    agent = CFRAgent(
        player_id=0,
        num_actions=num_actions,
        obs_size=obs_size,
        config=cfr_config,
        game=game,
    )
    
    print(f"\nCFR Configuration:")
    print(f"  Iterations: {cfr_config.num_iterations}")
    print(f"  Save path: {cfg.training.save_path}")
    
    # Run CFR training
    print("\n" + "-"*60)
    print("Starting CFR iterations...")
    print("-"*60 + "\n")
    
    metrics = agent.train()
    
    # Save the trained policy
    save_path = os.path.abspath(cfg.training.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    
    # Print final results
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final exploitability: {metrics['exploitability']:.6f}")
    print(f"  (0 = perfect Nash equilibrium)")
    print(f"\nModel saved to: {save_path}")
    print("\nYou can now evaluate this agent against DQN/PPO/Random agents")
    print("using the evaluation scripts.")
    print("="*60)
    
    return agent, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CFR agent for Leduc Hold'em poker"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cfr.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of CFR iterations (overrides config)"
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = ConfigObject(yaml.safe_load(f))
    
    # Override iterations if specified
    if args.iterations is not None:
        cfg.cfr.num_iterations = args.iterations
    
    # Train
    train_cfr(cfg)
