"""
run_full_evaluation.py — Comprehensive evaluation with visualizations.

This script runs a complete evaluation suite and generates visualizations
for comparing different poker agents (CFR, DQN, PPO, Random).

Usage:
    # Evaluate all agents with visualizations
    python run_full_evaluation.py
    
    # Evaluate specific agents
    python run_full_evaluation.py --agents cfr dqn ppo random
    
    # Custom number of episodes
    python run_full_evaluation.py --episodes 5000
    
    # Save to custom directory
    python run_full_evaluation.py --output-dir my_results
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np

from poker_env import PokerEnv
from evaluate import evaluate_head_to_head, evaluate_round_robin
from run_evaluation import make_agent
from visualization import (
    plot_head_to_head,
    plot_round_robin_heatmap,
    plot_round_robin_ranking,
    create_results_summary,
    save_results_to_csv,
)
from config import SEED


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive poker agent evaluation with visualizations"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["random", "cfr"],
        help="Agent names to evaluate (default: random cfr)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of episodes per matchup (default: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (only save data)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("COMPREHENSIVE POKER AGENT EVALUATION")
    print("=" * 70)
    print(f"Agents: {', '.join(args.agents)}")
    print(f"Episodes per matchup: {args.episodes}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    print()
    
    # Initialize environment
    env = PokerEnv(seed=args.seed)
    print(f"Environment: {env.game}")
    print(f"Players: {env.num_players}")
    print(f"Actions: {env.num_actions}")
    print(f"Observation size: {env.obs_size}")
    print()
    
    # ══════════════════════════════════════════════════════════
    # Part 1: Round-Robin Tournament
    # ══════════════════════════════════════════════════════════
    
    print("=" * 70)
    print("PART 1: ROUND-ROBIN TOURNAMENT")
    print("=" * 70)
    print()
    
    # Create agents
    agents_dict = {}
    for name in args.agents:
        print(f"Loading {name} agent...")
        try:
            agents_dict[name] = make_agent(name, player_id=0, env=env)
            print(f"  ✓ {name} loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load {name}: {e}")
            sys.exit(1)
    print()
    
    # Run round-robin
    print(f"Running round-robin tournament ({args.episodes} episodes per matchup)...")
    rr_result = evaluate_round_robin(agents_dict, num_episodes=args.episodes)
    print("✓ Tournament complete")
    print()
    
    # Print results
    print("PAYOFF MATRIX:")
    print("-" * 70)
    header = "          " + "  ".join(f"{name:>10s}" for name in rr_result.agent_names)
    print(header)
    print("-" * 70)
    for i, name in enumerate(rr_result.agent_names):
        row_vals = "  ".join(
            f"{rr_result.payoff_matrix[i, j]:>+10.4f}" if i != j else f"{'---':>10s}"
            for j in range(len(rr_result.agent_names))
        )
        print(f"{name:>10s}  {row_vals}")
    print()
    
    # Create summary table
    summary_df = create_results_summary(
        rr_result.agent_names,
        rr_result.payoff_matrix,
        save_path=os.path.join(output_dir, "tournament_summary.csv")
    )
    print("\nRANKINGS:")
    print("-" * 70)
    print(summary_df.to_string(index=False))
    print()
    
    # Generate visualizations
    if not args.no_plots:
        print("Generating tournament visualizations...")
        
        plot_round_robin_heatmap(
            rr_result.payoff_matrix,
            rr_result.agent_names,
            title=f"Round-Robin Tournament ({args.episodes} episodes per matchup)",
            save_path=os.path.join(output_dir, "tournament_heatmap.png"),
            show=False
        )
        
        plot_round_robin_ranking(
            rr_result.payoff_matrix,
            rr_result.agent_names,
            title="Agent Rankings (Mean Return vs All Opponents)",
            save_path=os.path.join(output_dir, "tournament_ranking.png"),
            show=False
        )
        
        print("✓ Tournament visualizations saved")
        print()
    
    # ══════════════════════════════════════════════════════════
    # Part 2: Detailed Head-to-Head Analyses
    # ══════════════════════════════════════════════════════════
    
    if len(args.agents) >= 2 and not args.no_plots:
        print("=" * 70)
        print("PART 2: DETAILED HEAD-TO-HEAD ANALYSES")
        print("=" * 70)
        print()
        
        # Find best and worst agents
        mean_payoffs = []
        for i in range(len(rr_result.agent_names)):
            payoffs = [rr_result.payoff_matrix[i, j] 
                      for j in range(len(rr_result.agent_names)) if i != j]
            mean_payoffs.append(np.mean(payoffs))
        
        best_idx = np.argmax(mean_payoffs)
        worst_idx = np.argmin(mean_payoffs)
        best_agent_name = rr_result.agent_names[best_idx]
        worst_agent_name = rr_result.agent_names[worst_idx]
        
        # Best vs Worst
        if best_agent_name != worst_agent_name:
            print(f"Analyzing: {best_agent_name} (best) vs {worst_agent_name} (worst)")
            
            agent_a = make_agent(best_agent_name, player_id=0, env=env)
            agent_b = make_agent(worst_agent_name, player_id=1, env=env)
            
            result = evaluate_head_to_head(
                agent_a, agent_b,
                num_episodes=args.episodes,
                env=env,
                agent_a_name=best_agent_name,
                agent_b_name=worst_agent_name,
            )
            
            # Collect returns for visualization
            env_temp = PokerEnv(seed=args.seed + 1000)
            returns_a = []
            returns_b = []
            
            from poker_env import play_episode
            for _ in range(args.episodes):
                ep_result = play_episode(env_temp, {0: agent_a, 1: agent_b})
                returns_a.append(ep_result.returns[0])
                returns_b.append(ep_result.returns[1])
            
            plot_head_to_head(
                best_agent_name,
                worst_agent_name,
                returns_a,
                returns_b,
                save_path=os.path.join(output_dir, f"h2h_{best_agent_name}_vs_{worst_agent_name}.png"),
                show=False
            )
            
            print(f"  Mean return ({best_agent_name}): {result.mean_return_a:+.4f}")
            print(f"  Mean return ({worst_agent_name}): {result.mean_return_b:+.4f}")
            print(f"  Win rate ({best_agent_name}): {result.win_rate_a:.1%}")
            print(f"✓ Analysis saved")
            print()
        
        # All pairwise comparisons (if not too many)
        if len(args.agents) <= 4:
            print("Generating all pairwise comparisons...")
            for i, name_a in enumerate(args.agents):
                for j, name_b in enumerate(args.agents):
                    if i >= j:
                        continue
                    
                    print(f"  {name_a} vs {name_b}...", end=" ")
                    
                    agent_a = make_agent(name_a, player_id=0, env=env)
                    agent_b = make_agent(name_b, player_id=1, env=env)
                    
                    # Collect returns
                    env_temp = PokerEnv(seed=args.seed + 2000 + i * 100 + j)
                    returns_a = []
                    returns_b = []
                    
                    from poker_env import play_episode
                    for _ in range(args.episodes):
                        ep_result = play_episode(env_temp, {0: agent_a, 1: agent_b})
                        returns_a.append(ep_result.returns[0])
                        returns_b.append(ep_result.returns[1])
                    
                    plot_head_to_head(
                        name_a,
                        name_b,
                        returns_a,
                        returns_b,
                        save_path=os.path.join(output_dir, f"h2h_{name_a}_vs_{name_b}.png"),
                        show=False
                    )
                    
                    print("✓")
            
            print("✓ All pairwise comparisons saved")
            print()
    
    # ══════════════════════════════════════════════════════════
    # Part 3: Save Raw Data
    # ══════════════════════════════════════════════════════════
    
    print("=" * 70)
    print("PART 3: SAVING RAW DATA")
    print("=" * 70)
    print()
    
    # Save payoff matrix
    payoff_data = {
        'Agent': rr_result.agent_names,
    }
    for j, opponent in enumerate(rr_result.agent_names):
        payoff_data[f'vs_{opponent}'] = rr_result.payoff_matrix[:, j]
    
    save_results_to_csv(
        payoff_data,
        os.path.join(output_dir, "payoff_matrix.csv")
    )
    
    # Save configuration
    config_path = os.path.join(output_dir, "config.txt")
    with open(config_path, 'w') as f:
        f.write("Evaluation Configuration\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Agents: {', '.join(args.agents)}\n")
        f.write(f"Episodes per matchup: {args.episodes}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Environment: {env.game}\n")
        f.write(f"Num actions: {env.num_actions}\n")
        f.write(f"Observation size: {env.obs_size}\n")
    print(f"✓ Configuration saved to {config_path}")
    
    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    
    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Results saved to: {output_dir}")
    print()
    print("Generated files:")
    print(f"  - tournament_summary.csv     (Rankings and statistics)")
    print(f"  - payoff_matrix.csv          (Full payoff matrix)")
    print(f"  - config.txt                 (Evaluation configuration)")
    
    if not args.no_plots:
        print(f"  - tournament_heatmap.png     (Payoff matrix heatmap)")
        print(f"  - tournament_ranking.png     (Agent rankings bar chart)")
        print(f"  - h2h_*.png                  (Head-to-head comparisons)")
    
    print()
    print("Top 3 Agents:")
    for i in range(min(3, len(summary_df))):
        row = summary_df.iloc[i]
        print(f"  {i+1}. {row['Agent']:>10s}  "
              f"(mean return: {row['Mean Return']:+.4f}, "
              f"wins: {int(row['Wins'])}/{len(args.agents)-1})")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
