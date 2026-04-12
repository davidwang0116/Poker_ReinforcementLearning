# Visualization Integration Examples for DQN/PPO

## 📘 For Person 2: How to Add Visualization to Your Training Code

This guide shows **exactly** how to integrate the visualization tools into your existing DQN and PPO training scripts.

## 🎯 Part 1: DQN Training with Visualization

### Step 1: Import Visualization Tools

Add to the top of your DQN training script:

```python
from visualization import plot_training_curves, plot_multi_agent_comparison
import os
```

### Step 2: Collect Metrics During Training

Modify your training loop to collect metrics:

```python
def train(self, env, num_episodes: int, **kwargs) -> dict:
    """Train DQN agent."""
    
    # Initialize metric tracking
    loss_history = []
    reward_history = []
    epsilon_history = []
    q_value_history = []
    
    for episode in range(num_episodes):
        # ... your existing training code ...
        
        # Collect metrics (add these lines)
        if hasattr(self, 'last_loss') and self.last_loss is not None:
            loss_history.append(self.last_loss)
        
        # Track episode reward
        episode_reward = 0  # Sum rewards during episode
        reward_history.append(episode_reward)
        
        # Track epsilon
        epsilon_history.append(self.epsilon)
        
        # Track average Q-value (optional)
        if hasattr(self, 'last_q_values'):
            q_value_history.append(self.last_q_values.mean())
        
        # Print progress
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Loss={loss_history[-1]:.4f}, "
                  f"Reward={reward_history[-1]:.2f}, "
                  f"Epsilon={epsilon_history[-1]:.3f}")
    
    # Return metrics for visualization
    return {
        'loss': loss_history,
        'reward': reward_history,
        'epsilon': epsilon_history,
        'q_values': q_value_history
    }
```

### Step 3: Visualize After Training

```python
# After training completes
metrics = agent.train(env, num_episodes=100000)

# Generate training curves
plot_training_curves(
    {
        'loss': metrics['loss'],
        'reward': metrics['reward'],
        'epsilon': metrics['epsilon']
    },
    title='DQN Training Progress (100k episodes)',
    save_path='results/dqn/training_curves.png',
    show=True
)

print("✓ Training visualization saved to results/dqn/training_curves.png")
```

### Complete DQN Example

```python
"""
Example: Add visualization to DQN training
"""

from agents.dqn_agent import DQNAgent
from poker_env import PokerEnv
from visualization import plot_training_curves
from config import DQNConfig

# Create agent and environment
env = PokerEnv()
config = DQNConfig()
agent = DQNAgent(player_id=0, num_actions=env.num_actions, 
                 obs_size=env.obs_size, config=config)

# Train with metric collection
print("Training DQN agent...")
metrics = agent.train(env, num_episodes=100000)

# Visualize
print("Generating visualizations...")
plot_training_curves(
    metrics,
    title='DQN Training Progress',
    save_path='results/dqn/training_curves.png',
    show=True
)

# Save model
agent.save('models/dqn/dqn_final.pth')

print("✓ Training complete!")
print(f"✓ Model saved to models/dqn/dqn_final.pth")
print(f"✓ Plots saved to results/dqn/")
```

## 🎯 Part 2: PPO Training with Visualization

### PPO-Specific Metrics

```python
def train(self, env, num_episodes: int, **kwargs) -> dict:
    """Train PPO agent."""
    
    # Track PPO-specific metrics
    policy_loss_history = []
    value_loss_history = []
    entropy_history = []
    reward_history = []
    kl_divergence_history = []  # Optional
    
    for episode in range(num_episodes):
        # ... your PPO training code ...
        
        # After each update, record metrics
        policy_loss_history.append(self.last_policy_loss)
        value_loss_history.append(self.last_value_loss)
        entropy_history.append(self.last_entropy)
        reward_history.append(episode_reward)
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode+1}: "
                  f"Policy Loss={policy_loss_history[-1]:.4f}, "
                  f"Value Loss={value_loss_history[-1]:.4f}, "
                  f"Reward={reward_history[-1]:.2f}")
    
    return {
        'policy_loss': policy_loss_history,
        'value_loss': value_loss_history,
        'entropy': entropy_history,
        'reward': reward_history
    }
```

### Visualize PPO Training

```python
# After training
metrics = agent.train(env, num_episodes=100000)

# Generate comprehensive plots
plot_training_curves(
    metrics,
    title='PPO Training Progress (100k episodes)',
    save_path='results/ppo/training_curves.png',
    show=True
)
```

## 🏆 Part 3: Compare Your Models Against CFR

### Simple Comparison

```bash
# Command-line (easiest)
python run_full_evaluation.py --agents random dqn ppo cfr --episodes 1000
```

### Programmatic Comparison

```python
"""
compare_all_agents.py - Compare DQN, PPO, and CFR
"""

from poker_env import PokerEnv, play_episode
from run_evaluation import make_agent
from visualization import plot_head_to_head, plot_round_robin_heatmap
from evaluate import evaluate_round_robin

# Load all agents
env = PokerEnv()
agents = {
    'Random': make_agent('random', player_id=0, env=env),
    'DQN': make_agent('dqn', player_id=0, env=env),
    'PPO': make_agent('ppo', player_id=0, env=env),
    'CFR': make_agent('cfr', player_id=0, env=env)
}

print("Running tournament...")
rr_result = evaluate_round_robin(agents, num_episodes=1000)

# Visualize results
plot_round_robin_heatmap(
    rr_result.payoff_matrix,
    rr_result.agent_names,
    title='Final Tournament Results',
    save_path='results/final_tournament.png',
    show=True
)

print("✓ Tournament complete!")
print("\nResults:")
for i, name in enumerate(rr_result.agent_names):
    mean_return = rr_result.payoff_matrix[i, :].mean()
    print(f"  {name}: mean return = {mean_return:+.4f}")
```

## 🎨 Part 4: Create Plots for Your Report

### Figure 1: Training Comparison

```python
from visualization import plot_multi_agent_comparison

# Combine all training metrics
agent_metrics = {
    'DQN': {
        'loss': dqn_metrics['loss'],
        'reward': dqn_metrics['reward']
    },
    'PPO': {
        'loss': ppo_metrics['policy_loss'],
        'reward': ppo_metrics['reward']
    },
    'CFR': {
        'exploitability': cfr_metrics['exploitability']
    }
}

plot_multi_agent_comparison(
    agent_metrics,
    title='Figure 1: Training Progress Comparison',
    save_path='results/paper/figure1_training.png',
    show=False
)
```

### Figure 2: Tournament Results

```python
from visualization import plot_round_robin_heatmap

plot_round_robin_heatmap(
    payoff_matrix,
    agent_names,
    title='Figure 2: Round-Robin Tournament Results',
    save_path='results/paper/figure2_tournament.png',
    show=False
)
```

### Figure 3: DQN vs CFR Detailed

```python
from visualization import plot_head_to_head

# Run DQN vs CFR
dqn_returns, cfr_returns = [], []
for _ in range(1000):
    result = play_episode(env, {0: dqn_agent, 1: cfr_agent})
    dqn_returns.append(result.returns[0])
    cfr_returns.append(result.returns[1])

plot_head_to_head(
    'DQN',
    'CFR',
    dqn_returns,
    cfr_returns,
    save_path='results/paper/figure3_dqn_vs_cfr.png',
    show=False
)
```

### Figure 4: PPO vs CFR Detailed

```python
# Similar to above, but PPO vs CFR
plot_head_to_head(
    'PPO',
    'CFR',
    ppo_returns,
    cfr_returns,
    save_path='results/paper/figure4_ppo_vs_cfr.png',
    show=False
)
```

## 📊 Part 5: Example Complete Analysis Script

Save this as `final_analysis.py`:

```python
"""
final_analysis.py - Generate all plots for final report
Run this after training all models.
"""

import os
from poker_env import PokerEnv, play_episode
from run_evaluation import make_agent
from evaluate import evaluate_round_robin
from visualization import (
    plot_training_curves,
    plot_multi_agent_comparison,
    plot_head_to_head,
    plot_round_robin_heatmap,
    plot_round_robin_ranking,
    create_results_summary
)

# Create output directory
os.makedirs('results/paper', exist_ok=True)

print("="*70)
print("FINAL ANALYSIS - Generating All Plots")
print("="*70)

# Load environment
env = PokerEnv()

# ═══════════════════════════════════════════════════════════
# 1. Training Curves (if you saved metrics during training)
# ═══════════════════════════════════════════════════════════

print("\n1. Training curves...")

# Load your saved metrics (you need to save these during training)
# Example:
# import pickle
# with open('results/dqn_metrics.pkl', 'rb') as f:
#     dqn_metrics = pickle.load(f)
# with open('results/ppo_metrics.pkl', 'rb') as f:
#     ppo_metrics = pickle.load(f)

# For now, using placeholder
# TODO: Replace with your actual metrics
print("   ⚠ Add your training metrics here")

# ═══════════════════════════════════════════════════════════
# 2. Round-Robin Tournament
# ═══════════════════════════════════════════════════════════

print("\n2. Running round-robin tournament...")

agents = {
    'Random': make_agent('random', player_id=0, env=env),
    'DQN': make_agent('dqn', player_id=0, env=env),
    'PPO': make_agent('ppo', player_id=0, env=env),
    'CFR': make_agent('cfr', player_id=0, env=env)
}

rr_result = evaluate_round_robin(agents, num_episodes=5000)

# Heatmap
plot_round_robin_heatmap(
    rr_result.payoff_matrix,
    rr_result.agent_names,
    title='Round-Robin Tournament Results (5000 episodes)',
    save_path='results/paper/tournament_heatmap.png',
    show=False
)
print("   ✓ Heatmap saved")

# Rankings
plot_round_robin_ranking(
    rr_result.payoff_matrix,
    rr_result.agent_names,
    title='Agent Rankings',
    save_path='results/paper/tournament_ranking.png',
    show=False
)
print("   ✓ Rankings saved")

# Summary table
summary_df = create_results_summary(
    rr_result.agent_names,
    rr_result.payoff_matrix,
    save_path='results/paper/tournament_summary.csv'
)
print("   ✓ Summary table saved")

# ═══════════════════════════════════════════════════════════
# 3. Detailed Head-to-Head Comparisons
# ═══════════════════════════════════════════════════════════

print("\n3. Generating head-to-head comparisons...")

# DQN vs CFR
print("   Running DQN vs CFR...")
dqn_agent = make_agent('dqn', player_id=0, env=env)
cfr_agent = make_agent('cfr', player_id=1, env=env)

returns_dqn, returns_cfr = [], []
for _ in range(1000):
    result = play_episode(env, {0: dqn_agent, 1: cfr_agent})
    returns_dqn.append(result.returns[0])
    returns_cfr.append(result.returns[1])

plot_head_to_head(
    'DQN', 'CFR',
    returns_dqn, returns_cfr,
    save_path='results/paper/dqn_vs_cfr.png',
    show=False
)
print("   ✓ DQN vs CFR saved")

# PPO vs CFR
print("   Running PPO vs CFR...")
ppo_agent = make_agent('ppo', player_id=0, env=env)

returns_ppo, returns_cfr2 = [], []
for _ in range(1000):
    result = play_episode(env, {0: ppo_agent, 1: cfr_agent})
    returns_ppo.append(result.returns[0])
    returns_cfr2.append(result.returns[1])

plot_head_to_head(
    'PPO', 'CFR',
    returns_ppo, returns_cfr2,
    save_path='results/paper/ppo_vs_cfr.png',
    show=False
)
print("   ✓ PPO vs CFR saved")

# ═══════════════════════════════════════════════════════════
# 4. Summary
# ═══════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"\nAll plots saved to: results/paper/")
print("\nGenerated files:")
print("  - tournament_heatmap.png")
print("  - tournament_ranking.png")
print("  - tournament_summary.csv")
print("  - dqn_vs_cfr.png")
print("  - ppo_vs_cfr.png")
print("\nFinal Rankings:")
print(summary_df.to_string(index=False))
```

Save this as `final_analysis.py` and run:
```bash
python final_analysis.py
```

## 🔧 Part 2: Modify Your Existing Training Scripts

### If you're using `main.py` for training:

Add this at the end of `main.py`:

```python
# At the end of main.py, after training
if __name__ == "__main__":
    # ... existing training code ...
    
    # Add visualization
    from visualization import plot_training_curves
    
    if hasattr(agent, 'training_metrics'):
        plot_training_curves(
            agent.training_metrics,
            title=f'{agent_type.upper()} Training',
            save_path=f'results/{agent_type}_training.png',
            show=False
        )
        print(f"✓ Training plot saved to results/{agent_type}_training.png")
```

### If you have separate training scripts:

Add to `train_dqn.py` or `train_ppo.py`:

```python
# After training
from visualization import plot_training_curves

metrics = agent.train(env, num_episodes=100000)

# Visualize
plot_training_curves(
    metrics,
    title='Training Progress',
    save_path='results/training.png',
    show=True
)
```

## 📈 Part 3: Real-Time Training Monitoring

### Option A: Periodic Plotting During Training

```python
from visualization import plot_training_curves

def train(self, env, num_episodes: int, **kwargs) -> dict:
    loss_history = []
    reward_history = []
    
    for episode in range(num_episodes):
        # ... training code ...
        
        loss_history.append(loss)
        reward_history.append(reward)
        
        # Plot every 10k episodes
        if (episode + 1) % 10000 == 0:
            plot_training_curves(
                {'loss': loss_history, 'reward': reward_history},
                title=f'Training Progress (Episode {episode+1})',
                save_path=f'results/training_ep{episode+1}.png',
                show=False
            )
            print(f"  ✓ Checkpoint plot saved")
    
    return {'loss': loss_history, 'reward': reward_history}
```

### Option B: Save Metrics, Plot Later

```python
import pickle

# During training - just save metrics
def train(self, env, num_episodes: int, **kwargs) -> dict:
    metrics = {'loss': [], 'reward': []}
    
    for episode in range(num_episodes):
        # ... training ...
        metrics['loss'].append(loss)
        metrics['reward'].append(reward)
    
    # Save metrics
    with open('results/training_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    return metrics

# Later - load and plot
with open('results/training_metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

from visualization import plot_training_curves
plot_training_curves(metrics, save_path='results/training.png')
```

## 🎯 Part 4: Competition Workflow

### Complete Competition Script

```python
"""
compete_against_cfr.py - Test your models against CFR
"""

from poker_env import PokerEnv, play_episode
from run_evaluation import make_agent
from visualization import plot_head_to_head, save_results_to_csv
from evaluate import evaluate_head_to_head

env = PokerEnv()

# Load agents
print("Loading agents...")
dqn = make_agent('dqn', player_id=0, env=env)
ppo = make_agent('ppo', player_id=0, env=env)
cfr = make_agent('cfr', player_id=1, env=env)
print("✓ All agents loaded")

# ═══════════════════════════════════════════════════════════
# Test 1: DQN vs CFR
# ═══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("DQN vs CFR")
print("="*60)

result_dqn = evaluate_head_to_head(
    dqn, cfr,
    num_episodes=1000,
    env=env,
    agent_a_name='DQN',
    agent_b_name='CFR'
)

print(result_dqn.summary())

# Collect detailed returns for visualization
returns_dqn, returns_cfr = [], []
for _ in range(1000):
    ep_result = play_episode(env, {0: dqn, 1: cfr})
    returns_dqn.append(ep_result.returns[0])
    returns_cfr.append(ep_result.returns[1])

plot_head_to_head(
    'DQN', 'CFR',
    returns_dqn, returns_cfr,
    save_path='results/competition/dqn_vs_cfr.png',
    show=True
)

# ═══════════════════════════════════════════════════════════
# Test 2: PPO vs CFR
# ═══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("PPO vs CFR")
print("="*60)

result_ppo = evaluate_head_to_head(
    ppo, cfr,
    num_episodes=1000,
    env=env,
    agent_a_name='PPO',
    agent_b_name='CFR'
)

print(result_ppo.summary())

returns_ppo, returns_cfr2 = [], []
for _ in range(1000):
    ep_result = play_episode(env, {0: ppo, 1: cfr})
    returns_ppo.append(ep_result.returns[0])
    returns_cfr2.append(ep_result.returns[1])

plot_head_to_head(
    'PPO', 'CFR',
    returns_ppo, returns_cfr2,
    save_path='results/competition/ppo_vs_cfr.png',
    show=True
)

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("COMPETITION SUMMARY")
print("="*60)

print(f"\nDQN vs CFR:")
print(f"  DQN mean return: {result_dqn.mean_return_a:+.4f}")
print(f"  DQN win rate: {result_dqn.win_rate_a:.1%}")

print(f"\nPPO vs CFR:")
print(f"  PPO mean return: {result_ppo.mean_return_a:+.4f}")
print(f"  PPO win rate: {result_ppo.win_rate_a:.1%}")

# Save summary
summary_data = {
    'Agent': ['DQN', 'PPO'],
    'Mean_Return_vs_CFR': [result_dqn.mean_return_a, result_ppo.mean_return_a],
    'Win_Rate_vs_CFR': [result_dqn.win_rate_a, result_ppo.win_rate_a],
    'Std_Dev': [result_dqn.std_return_a, result_ppo.std_return_a]
}

save_results_to_csv(summary_data, 'results/competition/summary.csv')
print("\n✓ Summary saved to results/competition/summary.csv")
```

## 📝 Part 5: Minimal Integration (If Short on Time)

If you just want quick plots without modifying your training code:

```bash
# After training your models, just run:
python run_full_evaluation.py --agents random dqn ppo cfr --episodes 5000

# This generates everything automatically!
```

## 🎓 Part 6: For Your Report

### Recommended Figures

1. **Figure 1**: Training curves comparison (DQN vs PPO vs CFR)
2. **Figure 2**: Round-robin tournament heatmap
3. **Figure 3**: Agent rankings bar chart
4. **Figure 4**: DQN vs CFR detailed analysis
5. **Figure 5**: PPO vs CFR detailed analysis

### Table 1: Tournament Summary

Use the CSV output from `tournament_summary.csv`:
```
Rank | Agent  | Mean Return | Win Rate | Std Dev
-----|--------|-------------|----------|--------
1    | CFR    | +1.234      | 66.7%    | 2.345
2    | PPO    | +0.567      | 55.6%    | 2.456
3    | DQN    | -0.234      | 44.4%    | 2.567
4    | Random | -1.567      | 33.3%    | 2.678
```

## ✅ Checklist

Before final submission:

- [ ] DQN model trained
- [ ] PPO model trained
- [ ] Both models evaluated vs Random (should win)
- [ ] Both models evaluated vs CFR
- [ ] Training curves generated
- [ ] Tournament heatmap generated
- [ ] Rankings chart generated
- [ ] Head-to-head plots generated
- [ ] Summary statistics exported to CSV
- [ ] All plots saved to `results/paper/`

## 🚀 Quick Commands Summary

```bash
# 1. Train your models
python main.py --config configs/dqn_selfplay.yaml
python main.py --config configs/ppo_selfplay.yaml

# 2. Generate all visualizations
python run_full_evaluation.py --agents random dqn ppo cfr --episodes 5000

# 3. Done! Check results/YYYYMMDD_HHMMSS/
```

## 💡 Pro Tips

1. **Save metrics during training**: Easier to plot later
2. **Use high episode counts**: 5000+ for stable statistics
3. **Generate plots early**: Don't wait until last minute
4. **Save raw data**: CSV files for backup
5. **Use descriptive titles**: Include parameters in plot titles

## 📞 Questions?

- Check `VISUALIZATION_GUIDE.md` for detailed examples
- Run `python visualization.py` to see example plots
- Ask Person 3 (CFR author) for help

Good luck with your DQN/PPO implementation! 🎉
