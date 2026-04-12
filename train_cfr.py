"""
train_cfr.py — Train a CFR agent on Leduc Hold'em poker.

This script trains a Counterfactual Regret Minimization (CFR) agent
to compute a Nash equilibrium strategy for Leduc poker. Unlike RL
agents (DQN, PPO), CFR does not learn from experience replay or
policy gradients. Instead, it iteratively traverses the full game
tree, accumulating regret for each action at each information set.

Usage:
    python train_cfr.py [--iterations N] [--save-path PATH]

Example:
    # Train for 10,000 iterations (default)
    python train_cfr.py
    
    # Train for 100,000 iterations for better convergence
    python train_cfr.py --iterations 100000
    
    # Save to custom path
    python train_cfr.py --save-path models/cfr/my_cfr_agent.pkl
"""

import argparse
import os
import pyspiel
from poker_env import PokerEnv
from agents.cfr_agent import CFRAgent
from agents.random_agent import RandomAgent
from evaluate import evaluate_head_to_head
from config import CFRConfig
from visualization import plot_exploitability_convergence, plot_training_curves


def main():
    parser = argparse.ArgumentParser(
        description="Train a CFR agent on Leduc Hold'em poker"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of CFR iterations (default: 10000)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/cfr/cfr_agent.pkl",
        help="Path to save the trained agent (default: models/cfr/cfr_agent.pkl)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=1000,
        help="Number of episodes for evaluation (default: 1000)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate training plots (default: False)",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="results/cfr_training",
        help="Directory to save plots (default: results/cfr_training)",
    )
    args = parser.parse_args()
    
    print("="*60)
    print("CFR Agent Training")
    print("="*60)
    print(f"Iterations: {args.iterations}")
    print(f"Save path: {args.save_path}")
    print()
    
    # Create environment and game
    env = PokerEnv()
    game = pyspiel.load_game("leduc_poker")
    
    print(f"Game: {game.get_type().short_name}")
    print(f"Num actions: {env.num_actions}")
    print(f"Observation size: {env.obs_size}")
    print()
    
    # Create CFR agent
    config = CFRConfig(num_iterations=args.iterations)
    cfr_agent = CFRAgent(
        player_id=0,
        num_actions=env.num_actions,
        game=game,
        config=config,
    )
    
    # Train
    print("="*60)
    print("Training")
    print("="*60)
    
    metrics = cfr_agent.train(env)
    
    print()
    print("="*60)
    print("Training Summary")
    print("="*60)
    print(f"Iterations: {metrics['iterations']}")
    print(f"Final exploitability: {metrics['final_exploitability']:.6f}")
    print(f"  (Lower is better; 0 = perfect Nash equilibrium)")
    print()
    
    # Evaluate against random agent
    print("="*60)
    print("Evaluation vs Random Agent")
    print("="*60)
    
    random_agent = RandomAgent(player_id=1, num_actions=env.num_actions)
    
    result = evaluate_head_to_head(
        cfr_agent,
        random_agent,
        num_episodes=args.eval_episodes,
        env=env,
        agent_a_name="CFR",
        agent_b_name="Random",
    )
    
    print(result.summary())
    
    # Generate plots if requested
    if args.plot:
        print()
        print("="*60)
        print("Generating Visualizations")
        print("="*60)
        
        os.makedirs(args.plot_dir, exist_ok=True)
        
        # Plot exploitability convergence
        if 'exploitability' in metrics and len(metrics['exploitability']) > 0:
            # Calculate iteration numbers for checkpoints
            total_iters = metrics['iterations']
            num_checkpoints = len(metrics['exploitability'])
            checkpoint_interval = total_iters // num_checkpoints
            iterations = [checkpoint_interval * (i + 1) for i in range(num_checkpoints)]
            
            plot_exploitability_convergence(
                metrics['exploitability'],
                iterations=iterations,
                title=f"CFR Exploitability Convergence ({total_iters} iterations)",
                save_path=os.path.join(args.plot_dir, "exploitability_convergence.png"),
                show=False
            )
            print(f"  ✓ Exploitability plot saved")
        
        # Plot training metrics
        plot_metrics = {
            'Exploitability': metrics['exploitability'],
        }
        plot_training_curves(
            plot_metrics,
            title=f"CFR Training Metrics ({metrics['iterations']} iterations)",
            save_path=os.path.join(args.plot_dir, "training_curves.png"),
            show=False
        )
        print(f"  ✓ Training curves saved")
        print(f"  Plots saved to: {args.plot_dir}")
    
    # Save the agent
    print()
    print("="*60)
    print("Saving Agent")
    print("="*60)
    
    # Ensure directory exists
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    cfr_agent.save(args.save_path)
    
    # Final summary
    print()
    print("="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved to: {args.save_path}")
    print(f"Final exploitability: {metrics['final_exploitability']:.6f}")
    print(f"Win rate vs Random: {result.win_rate_a:.1%}")
    print(f"Mean return vs Random: {result.mean_return_a:+.4f}")
    print()
    
    if metrics['final_exploitability'] < 0.1:
        print("✓ Excellent convergence! Exploitability < 0.1")
    elif metrics['final_exploitability'] < 0.5:
        print("✓ Good convergence! Exploitability < 0.5")
    else:
        print("⚠ Consider training for more iterations for better convergence")
    
    if result.mean_return_a > 0:
        print("✓ CFR agent is beating random agent")
    else:
        print("⚠ CFR agent performance could be improved")


if __name__ == "__main__":
    main()
