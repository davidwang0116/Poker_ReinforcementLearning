"""
visualization.py — Visualization utilities for poker RL experiments.

This module provides plotting functions for analyzing and comparing
different poker agents (CFR, DQN, PPO, Random). All functions work
with standard Python data structures and can be used by any agent.

Usage:
    from visualization import plot_training_curves, plot_head_to_head, ...
    
    # Plot training metrics
    plot_training_curves(metrics_dict, save_path='results/training.png')
    
    # Compare agents
    plot_round_robin_heatmap(payoff_matrix, agent_names, save_path='results/comparison.png')
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ══════════════════════════════════════════════════════════
# Training Curves
# ══════════════════════════════════════════════════════════

def plot_training_curves(metrics, title="Training Progress", save_path=None, show=True, window=1000):
    """
    绘制训练曲线，并对 Reward 进行平滑处理。
    :param metrics: 包含 'loss', 'reward', 'epsilon' 等列表的字典
    :param window: 滑动窗口大小，1000 代表计算最近 1000 局的平均值
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics), sharex=True)
    if num_metrics == 1: axes = [axes]

    for ax, (name, values) in zip(axes, metrics.items()):
        if name.lower() == 'reward':
            # --- 关键修改：计算滑动平均 ---
            # 使用 pandas 可以方便地处理滑动窗口
            series = pd.Series(values)
            smooth_values = series.rolling(window=window, min_periods=window//10).mean()
            
            # 绘制原始数据（浅色背景）
            ax.plot(values, alpha=0.2, color='orange', label='Raw Reward')
            # 绘制平滑曲线（深色主线）
            ax.plot(smooth_values, color='red', linewidth=2, label=f'MA (window={window})')
            ax.set_ylabel("Reward")
            ax.legend()
        
        elif name.lower() == 'loss':
            # Loss 通常也建议平滑，否则后期震荡太大会掩盖趋势
            series = pd.Series(values)
            smooth_loss = series.rolling(window=window//2, min_periods=10).mean()
            ax.plot(values, alpha=0.3, color='blue')
            ax.plot(smooth_loss, color='darkblue', linewidth=1.5)
            ax.set_ylabel("Loss (Log Scale recommended)")
            ax.set_yscale('log') # 建议开启对数坐标，因为你的 Loss 后来爆发了
            
        else:
            ax.plot(values)
            ax.set_ylabel(name.capitalize())

        ax.set_title(f"{title} - {name}")
        ax.grid(True, alpha=0.3)

    plt.xlabel("Episodes")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_exploitability_convergence(
    exploitability_values: List[float],
    iterations: Optional[List[int]] = None,
    title: str = "CFR Exploitability Convergence",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Figure:
    """Plot exploitability convergence for CFR agents.
    
    Parameters
    ----------
    exploitability_values : list of float
        Exploitability values at each checkpoint.
    iterations : list of int, optional
        Iteration numbers corresponding to each value.
        If None, uses sequential indices.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the plot.
    
    Returns
    -------
    Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if iterations is None:
        iterations = list(range(len(exploitability_values)))
    
    ax.plot(iterations, exploitability_values, 
            linewidth=2, marker='o', markersize=6, color='#2E86AB')
    
    # Add reference lines
    ax.axhline(y=0.1, color='orange', linestyle='--', linewidth=1.5, 
               label='Excellent (< 0.1)', alpha=0.7)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, 
               label='Good (< 0.5)', alpha=0.7)
    ax.axhline(y=0.0, color='green', linestyle='-', linewidth=1.5, 
               label='Nash Equilibrium (0.0)', alpha=0.5)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Exploitability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Log scale if values span multiple orders of magnitude
    if max(exploitability_values) / min(exploitability_values) > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Exploitability (log scale)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ══════════════════════════════════════════════════════════
# Head-to-Head Comparisons
# ══════════════════════════════════════════════════════════

def plot_head_to_head(
    agent_a_name: str,
    agent_b_name: str,
    agent_a_returns: List[float],
    agent_b_returns: List[float],
    save_path: Optional[str] = None,
    show: bool = True,
) -> Figure:
    """Plot head-to-head comparison between two agents.
    
    Parameters
    ----------
    agent_a_name : str
        Name of agent A.
    agent_b_name : str
        Name of agent B.
    agent_a_returns : list of float
        Returns for agent A across episodes.
    agent_b_returns : list of float
        Returns for agent B across episodes.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the plot.
    
    Returns
    -------
    Figure
        The matplotlib figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Return distributions (histograms)
    ax = axes[0, 0]
    ax.hist(agent_a_returns, bins=30, alpha=0.6, label=agent_a_name, color='#2E86AB')
    ax.hist(agent_b_returns, bins=30, alpha=0.6, label=agent_b_name, color='#A23B72')
    ax.set_xlabel('Return', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Return Distributions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Cumulative returns over episodes
    ax = axes[0, 1]
    cumulative_a = np.cumsum(agent_a_returns)
    cumulative_b = np.cumsum(agent_b_returns)
    episodes = list(range(1, len(agent_a_returns) + 1))
    ax.plot(episodes, cumulative_a, label=agent_a_name, linewidth=2, color='#2E86AB')
    ax.plot(episodes, cumulative_b, label=agent_b_name, linewidth=2, color='#A23B72')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Cumulative Return', fontsize=11)
    ax.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Win/Loss/Draw breakdown
    ax = axes[1, 0]
    wins_a = sum(1 for r in agent_a_returns if r > 0)
    wins_b = sum(1 for r in agent_b_returns if r > 0)
    draws = sum(1 for r in agent_a_returns if r == 0)
    
    categories = [agent_a_name, 'Draws', agent_b_name]
    values = [wins_a, draws, wins_b]
    colors = ['#2E86AB', '#F18F01', '#A23B72']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Win/Draw/Loss Breakdown', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_a = {
        'Mean Return': np.mean(agent_a_returns),
        'Std Dev': np.std(agent_a_returns),
        'Win Rate': wins_a / len(agent_a_returns),
        'Min Return': np.min(agent_a_returns),
        'Max Return': np.max(agent_a_returns),
    }
    
    stats_b = {
        'Mean Return': np.mean(agent_b_returns),
        'Std Dev': np.std(agent_b_returns),
        'Win Rate': wins_b / len(agent_b_returns),
        'Min Return': np.min(agent_b_returns),
        'Max Return': np.max(agent_b_returns),
    }
    
    summary_text = f"Summary Statistics\n{'='*40}\n\n"
    summary_text += f"{agent_a_name}:\n"
    for key, val in stats_a.items():
        if 'Rate' in key:
            summary_text += f"  {key}: {val:.1%}\n"
        else:
            summary_text += f"  {key}: {val:+.4f}\n"
    
    summary_text += f"\n{agent_b_name}:\n"
    for key, val in stats_b.items():
        if 'Rate' in key:
            summary_text += f"  {key}: {val:.1%}\n"
        else:
            summary_text += f"  {key}: {val:+.4f}\n"
    
    summary_text += f"\nTotal Episodes: {len(agent_a_returns)}"
    summary_text += f"\nDraws: {draws} ({draws/len(agent_a_returns):.1%})"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle(f'{agent_a_name} vs {agent_b_name}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ══════════════════════════════════════════════════════════
# Round-Robin Tournament Visualization
# ══════════════════════════════════════════════════════════

def plot_round_robin_heatmap(
    payoff_matrix: np.ndarray,
    agent_names: List[str],
    title: str = "Round-Robin Tournament Results",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Figure:
    """Plot heatmap of round-robin tournament results.
    
    Parameters
    ----------
    payoff_matrix : np.ndarray
        Square matrix where [i,j] = mean return of agent i vs agent j.
    agent_names : list of str
        Names of agents (row/column labels).
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the plot.
    
    Returns
    -------
    Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(payoff_matrix, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(agent_names)))
    ax.set_yticks(np.arange(len(agent_names)))
    ax.set_xticklabels(agent_names, fontsize=11)
    ax.set_yticklabels(agent_names, fontsize=11)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Return', rotation=270, labelpad=20, fontsize=11)
    
    # Add text annotations
    for i in range(len(agent_names)):
        for j in range(len(agent_names)):
            if i == j:
                text = ax.text(j, i, '---', ha="center", va="center", 
                              color="black", fontsize=10, fontweight='bold')
            else:
                value = payoff_matrix[i, j]
                color = "white" if abs(value) > 2 else "black"
                text = ax.text(j, i, f'{value:+.2f}', ha="center", va="center",
                              color=color, fontsize=10)
    
    ax.set_xlabel('Opponent', fontsize=12, fontweight='bold')
    ax.set_ylabel('Agent', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_round_robin_ranking(
    payoff_matrix: np.ndarray,
    agent_names: List[str],
    title: str = "Agent Rankings",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Figure:
    """Plot bar chart of agent rankings from round-robin tournament.
    
    Parameters
    ----------
    payoff_matrix : np.ndarray
        Square matrix where [i,j] = mean return of agent i vs agent j.
    agent_names : list of str
        Names of agents.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the plot.
    
    Returns
    -------
    Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute mean payoff for each agent (excluding self-play)
    n = len(agent_names)
    mean_payoffs = []
    for i in range(n):
        payoffs = [payoff_matrix[i, j] for j in range(n) if i != j]
        mean_payoffs.append(np.mean(payoffs))
    
    # Sort by performance
    sorted_indices = np.argsort(mean_payoffs)[::-1]
    sorted_names = [agent_names[i] for i in sorted_indices]
    sorted_payoffs = [mean_payoffs[i] for i in sorted_indices]
    
    # Create bar chart
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_names)))
    bars = ax.barh(sorted_names, sorted_payoffs, color=colors, alpha=0.8)
    
    # Add value labels
    for i, (bar, payoff) in enumerate(zip(bars, sorted_payoffs)):
        width = bar.get_width()
        label_x = width + 0.05 if width > 0 else width - 0.05
        ha = 'left' if width > 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{payoff:+.3f}',
                ha=ha, va='center', fontsize=11, fontweight='bold')
    
    # Add rank labels
    for i, name in enumerate(sorted_names):
        ax.text(-0.02, i, f'#{i+1}', ha='right', va='center',
                fontsize=11, fontweight='bold', 
                transform=ax.get_yaxis_transform())
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Mean Return vs Opponents', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ══════════════════════════════════════════════════════════
# Multi-Agent Comparison
# ══════════════════════════════════════════════════════════

def plot_multi_agent_comparison(
    agent_metrics: Dict[str, Dict[str, List[float]]],
    title: str = "Multi-Agent Training Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Figure:
    """Compare training curves across multiple agents.
    
    Parameters
    ----------
    agent_metrics : dict
        Nested dict: {agent_name: {metric_name: [values]}}.
        Example: {
            'DQN': {'loss': [1.0, 0.8], 'reward': [0.1, 0.3]},
            'PPO': {'loss': [0.9, 0.7], 'reward': [0.2, 0.4]}
        }
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the plot.
    
    Returns
    -------
    Figure
        The matplotlib figure object.
    """
    # Get all unique metrics
    all_metrics = set()
    for agent_data in agent_metrics.values():
        all_metrics.update(agent_data.keys())
    all_metrics = sorted(all_metrics)
    
    n_metrics = len(all_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
    
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_metrics)))
    
    for ax, metric_name in zip(axes, all_metrics):
        for (agent_name, agent_data), color in zip(agent_metrics.items(), colors):
            if metric_name in agent_data:
                values = agent_data[metric_name]
                ax.plot(values, label=agent_name, linewidth=2, 
                       marker='o', markersize=4, color=color, alpha=0.8)
        
        ax.set_xlabel('Iteration / Episode', fontsize=11)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric_name.replace("_", " ").title()} Comparison', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ══════════════════════════════════════════════════════════
# Utility Functions
# ══════════════════════════════════════════════════════════

def save_results_to_csv(
    data: Dict[str, List],
    filepath: str,
) -> None:
    """Save results to CSV file.
    
    Parameters
    ----------
    data : dict
        Dictionary mapping column names to lists of values.
    filepath : str
        Path to save CSV file.
    """
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✓ Saved results to {filepath}")


def create_results_summary(
    agent_names: List[str],
    payoff_matrix: np.ndarray,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Create a summary table of tournament results.
    
    Parameters
    ----------
    agent_names : list of str
        Names of agents.
    payoff_matrix : np.ndarray
        Payoff matrix from round-robin tournament.
    save_path : str, optional
        Path to save summary as CSV.
    
    Returns
    -------
    pd.DataFrame
        Summary table with rankings and statistics.
    """
    n = len(agent_names)
    
    # Compute statistics for each agent
    stats = []
    for i, name in enumerate(agent_names):
        payoffs = [payoff_matrix[i, j] for j in range(n) if i != j]
        stats.append({
            'Agent': name,
            'Mean Return': np.mean(payoffs),
            'Std Dev': np.std(payoffs),
            'Min Return': np.min(payoffs),
            'Max Return': np.max(payoffs),
            'Wins': sum(1 for p in payoffs if p > 0),
            'Losses': sum(1 for p in payoffs if p < 0),
            'Draws': sum(1 for p in payoffs if p == 0),
        })
    
    df = pd.DataFrame(stats)
    df = df.sort_values('Mean Return', ascending=False)
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✓ Saved summary to {save_path}")
    
    return df


# ══════════════════════════════════════════════════════════
# Example Usage
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Visualization Module - Example Usage")
    print("=" * 60)
    
    # Example 1: Training curves
    print("\n1. Training Curves Example")
    metrics = {
        'exploitability': [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
        'mean_reward': [0.1, 0.2, 0.3, 0.4, 0.45, 0.48, 0.5],
    }
    plot_training_curves(metrics, title='CFR Training', 
                        save_path='results/example_training.png', show=False)
    
    # Example 2: Head-to-head
    print("\n2. Head-to-Head Example")
    np.random.seed(42)
    returns_a = np.random.normal(0.5, 2, 100)
    returns_b = np.random.normal(-0.5, 2, 100)
    plot_head_to_head('CFR', 'Random', returns_a, returns_b,
                     save_path='results/example_h2h.png', show=False)
    
    # Example 3: Round-robin
    print("\n3. Round-Robin Example")
    agent_names = ['Random', 'DQN', 'PPO', 'CFR']
    payoff_matrix = np.array([
        [0, -2, -3, -4],
        [2, 0, -1, -2],
        [3, 1, 0, -0.5],
        [4, 2, 0.5, 0]
    ])
    plot_round_robin_heatmap(payoff_matrix, agent_names,
                            save_path='results/example_heatmap.png', show=False)
    plot_round_robin_ranking(payoff_matrix, agent_names,
                            save_path='results/example_ranking.png', show=False)
    
    # Example 4: Summary table
    print("\n4. Summary Table Example")
    summary = create_results_summary(agent_names, payoff_matrix,
                                    save_path='results/example_summary.csv')
    print("\n", summary)
    
    print("\n" + "=" * 60)
    print("✓ All example visualizations created in results/ directory")
