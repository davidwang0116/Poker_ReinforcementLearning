# Partner Guide: Using CFR Implementation & Visualization Tools

## 👋 Welcome!

This guide will help you (Person 2 - DQN/PPO) use the CFR implementation and visualization tools for your models.

## 📋 Quick Start

### 1. Pull the CFR Implementation

```bash
# Pull the feature branch
git fetch origin
git checkout feature/cfr-implementation

# Activate virtual environment
source .venv/bin/activate

# Install any new dependencies
pip install -r requirements.txt
```

### 2. Verify Everything Works

```bash
# Test the trained CFR model
python run_evaluation.py --mode exploitability --agents cfr

# Expected output: Exploitability ~0.096 (good quality)
```

## 🎯 Part 1: Using the Trained CFR Model

### What's Included

A pre-trained CFR model is available at:
- **Path**: `models/cfr/cfr_production.pkl`
- **Quality**: Exploitability ~0.096 (good, close to Nash equilibrium)
- **Size**: ~300KB
- **Training**: Trained with CFR algorithm

### Quick Test: CFR vs Random

```bash
# Evaluate CFR against random agent
python run_evaluation.py --mode head_to_head --agents cfr random --episodes 1000

# Expected: CFR should win consistently (mean return > 0)
```

### Compare Your DQN/PPO Against CFR

```bash
# DQN vs CFR
python run_evaluation.py --mode head_to_head --agents dqn cfr --episodes 1000

# PPO vs CFR
python run_evaluation.py --mode head_to_head --agents ppo cfr --episodes 1000

# Full tournament (all agents)
python run_evaluation.py --mode round_robin --agents random dqn ppo cfr --episodes 1000
```

## 📊 Part 2: Using Visualization Tools

The visualization module (`visualization.py`) provides comprehensive plotting tools for analyzing your DQN/PPO training and comparing agents.

### A. Visualize Your Training Progress

#### For DQN Training

Add this to your DQN training script:

```python
from visualization import plot_training_curves

# During training, collect metrics
loss_history = []
reward_history = []
epsilon_history = []

for episode in range(num_episodes):
    # ... your training code ...
    loss_history.append(loss)
    reward_history.append(episode_reward)
    epsilon_history.append(epsilon)

# After training, visualize
metrics = {
    'loss': loss_history,
    'reward': reward_history,
    'epsilon': epsilon_history
}

plot_training_curves(
    metrics,
    title='DQN Training Progress',
    save_path='results/dqn_training.png',
    show=True
)
```

#### For PPO Training

```python
from visualization import plot_training_curves

# Collect PPO-specific metrics
policy_loss_history = []
value_loss_history = []
reward_history = []
entropy_history = []

for episode in range(num_episodes):
    # ... your training code ...
    policy_loss_history.append(policy_loss)
    value_loss_history.append(value_loss)
    reward_history.append(episode_reward)
    entropy_history.append(entropy)

# Visualize
metrics = {
    'policy_loss': policy_loss_history,
    'value_loss': value_loss_history,
    'reward': reward_history,
    'entropy': entropy_history
}

plot_training_curves(
    metrics,
    title='PPO Training Progress',
    save_path='results/ppo_training.png',
    show=True
)
```

### B. Compare DQN vs PPO Training

```python
from visualization import plot_multi_agent_comparison

# After training both agents
agent_metrics = {
    'DQN': {
        'loss': dqn_loss_history,
        'reward': dqn_reward_history
    },
    'PPO': {
        'loss': ppo_loss_history,
        'reward': ppo_reward_history
    }
}

plot_multi_agent_comparison(
    agent_metrics,
    title='DQN vs PPO Training Comparison',
    save_path='results/dqn_vs_ppo_training.png',
    show=True
)
```

### C. Detailed Head-to-Head Analysis

```python
from visualization import plot_head_to_head
from poker_env import PokerEnv, play_episode
from run_evaluation import make_agent

# Load agents
env = PokerEnv()
dqn_agent = make_agent('dqn', player_id=0, env=env)
ppo_agent = make_agent('ppo', player_id=1, env=env)

# Run episodes and collect returns
returns_dqn = []
returns_ppo = []

for _ in range(1000):
    result = play_episode(env, {0: dqn_agent, 1: ppo_agent})
    returns_dqn.append(result.returns[0])
    returns_ppo.append(result.returns[1])

# Create detailed comparison plot
plot_head_to_head(
    'DQN',
    'PPO',
    returns_dqn,
    returns_ppo,
    save_path='results/dqn_vs_ppo_detailed.png',
    show=True
)
```

This creates a 4-panel plot showing:
1. Return distributions (histograms)
2. Cumulative returns over episodes
3. Win/Draw/Loss breakdown
4. Summary statistics

### D. Full Tournament with All Agents

Use the automated evaluation script:

```bash
# Run complete evaluation with visualizations
python run_full_evaluation.py --agents random dqn ppo cfr --episodes 1000

# Output saved to: results/YYYYMMDD_HHMMSS/
# Includes:
#   - tournament_heatmap.png (who beats who)
#   - tournament_ranking.png (overall rankings)
#   - h2h_*.png (detailed comparisons)
#   - tournament_summary.csv (statistics)
```

### E. Export Data for Further Analysis

```python
from visualization import save_results_to_csv, create_results_summary

# Save episode-by-episode data
data = {
    'episode': list(range(1, 1001)),
    'dqn_return': dqn_returns,
    'ppo_return': ppo_returns,
    'cfr_return': cfr_returns
}
save_results_to_csv(data, 'results/episode_data.csv')

# Create summary table
summary_df = create_results_summary(
    agent_names=['Random', 'DQN', 'PPO', 'CFR'],
    payoff_matrix=payoff_matrix,
    save_path='results/tournament_summary.csv'
)
print(summary_df)
```

## 🏆 Part 3: Complete Competition Workflow

### Step 1: Train Your Models

```bash
# Train DQN
python main.py --config configs/dqn_selfplay.yaml

# Train PPO
python main.py --config configs/ppo_selfplay.yaml
```

### Step 2: Quick Verification

```bash
# Test DQN vs Random
python run_evaluation.py --mode head_to_head --agents dqn random --episodes 100

# Test PPO vs Random
python run_evaluation.py --mode head_to_head --agents ppo random --episodes 100
```

### Step 3: Compare Against CFR

```bash
# DQN vs CFR (1000 episodes)
python run_evaluation.py --mode head_to_head --agents dqn cfr --episodes 1000

# PPO vs CFR (1000 episodes)
python run_evaluation.py --mode head_to_head --agents ppo cfr --episodes 1000
```

### Step 4: Full Tournament

```bash
# Run complete tournament with visualizations
python run_full_evaluation.py --agents random dqn ppo cfr --episodes 5000 --output-dir results/final_tournament
```

This generates:
- 📊 Heatmap showing all matchups
- 📈 Rankings bar chart
- 📉 Detailed head-to-head plots
- 📋 CSV files with statistics

### Step 5: Analyze Results

Check the output directory:
```
results/final_tournament/YYYYMMDD_HHMMSS/
├── tournament_heatmap.png      # Color-coded results matrix
├── tournament_ranking.png      # Overall agent rankings
├── h2h_DQN_vs_CFR.png         # Detailed DQN vs CFR
├── h2h_PPO_vs_CFR.png         # Detailed PPO vs CFR
├── tournament_summary.csv      # Statistics table
└── payoff_matrix.csv          # Raw results
```

## 📈 Part 4: Creating Publication-Quality Plots

### Example: Complete Analysis Script

```python
"""
complete_analysis.py - Generate all plots for final report
"""

from visualization import (
    plot_training_curves,
    plot_multi_agent_comparison,
    plot_head_to_head,
    plot_round_robin_heatmap,
    plot_round_robin_ranking,
    create_results_summary
)
from poker_env import PokerEnv, play_episode
from run_evaluation import make_agent
import numpy as np

# 1. Training curves comparison
agent_metrics = {
    'DQN': {
        'loss': dqn_loss_history,
        'reward': dqn_reward_history,
    },
    'PPO': {
        'loss': ppo_loss_history,
        'reward': ppo_reward_history,
    },
    'CFR': {
        'exploitability': cfr_exploit_history,
    }
}

plot_multi_agent_comparison(
    agent_metrics,
    title='Training Progress: DQN vs PPO vs CFR',
    save_path='results/paper/training_comparison.png',
    show=False
)

# 2. Round-robin tournament
env = PokerEnv()
agents = {
    'Random': make_agent('random', 0, env),
    'DQN': make_agent('dqn', 0, env),
    'PPO': make_agent('ppo', 0, env),
    'CFR': make_agent('cfr', 0, env)
}

# Run tournament
from evaluate import evaluate_round_robin
rr_result = evaluate_round_robin(agents, num_episodes=5000)

# Create visualizations
plot_round_robin_heatmap(
    rr_result.payoff_matrix,
    rr_result.agent_names,
    title='Round-Robin Tournament Results (5000 episodes per matchup)',
    save_path='results/paper/tournament_heatmap.png',
    show=False
)

plot_round_robin_ranking(
    rr_result.payoff_matrix,
    rr_result.agent_names,
    title='Final Agent Rankings',
    save_path='results/paper/tournament_ranking.png',
    show=False
)

# 3. Create summary table
summary_df = create_results_summary(
    rr_result.agent_names,
    rr_result.payoff_matrix,
    save_path='results/paper/final_summary.csv'
)

print("All plots generated in results/paper/")
print("\nFinal Rankings:")
print(summary_df)
```

## 🎨 Part 5: Customization Tips

### Change Plot Style

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Use different style
plt.style.use('ggplot')  # or 'seaborn', 'bmh', etc.

# Use different color palette
sns.set_palette("Set2")  # or "husl", "bright", etc.
```

### Adjust Figure Sizes

```python
# In your plotting code
fig, ax = plt.subplots(figsize=(12, 8))  # Larger figure
```

### Save High-Resolution Plots

```python
plt.savefig('results/my_plot.png', dpi=300, bbox_inches='tight')
```

## 🔧 Part 6: Troubleshooting

### Issue: "No module named 'seaborn'"

```bash
source .venv/bin/activate
pip install 'seaborn>=0.12'
```

### Issue: "CFR model not found"

Make sure you're using the correct path:
```python
# Correct
agent = make_agent('cfr', player_id=0, env=env, load_path='models/cfr/cfr_production.pkl')

# Or let it use default
agent = make_agent('cfr', player_id=0, env=env)
```

### Issue: Plots don't show

```python
# Add show=True
plot_training_curves(metrics, show=True)

# Or explicitly show
import matplotlib.pyplot as plt
plt.show()
```

### Issue: "Agent not trained"

Make sure your DQN/PPO models are saved in the correct locations:
- DQN: `models/dqn/dqn_selfplay_100w_final.pth`
- PPO: `models/ppo/ppo_selfplay_final.pth`

Or specify custom paths:
```python
agent = make_agent('dqn', player_id=0, env=env, load_path='models/dqn/my_model.pth')
```

## 📚 Part 7: Available Visualization Functions

| Function | Purpose | Output |
|----------|---------|--------|
| `plot_training_curves()` | Plot training metrics over time | Line plots |
| `plot_exploitability_convergence()` | CFR Nash convergence | Line plot with thresholds |
| `plot_head_to_head()` | Detailed 2-agent comparison | 4-panel figure |
| `plot_round_robin_heatmap()` | Tournament results matrix | Heatmap |
| `plot_round_robin_ranking()` | Overall rankings | Bar chart |
| `plot_multi_agent_comparison()` | Compare training curves | Multi-line plots |
| `save_results_to_csv()` | Export data | CSV file |
| `create_results_summary()` | Summary statistics | DataFrame/CSV |

## 📖 Part 8: Documentation

For more details, see:
- **Visualization Guide**: `VISUALIZATION_GUIDE.md` (detailed examples)
- **Visualization README**: `VISUALIZATION_README.md` (quick reference)
- **CFR README**: `CFR_README.md` (CFR-specific info)

## 🎯 Expected Results

Based on the CFR implementation:

### CFR Performance
- **Exploitability**: ~0.096 (good, close to Nash)
- **vs Random**: Mean return +0.3 to +0.7
- **Win Rate**: 45-55%

### Your Goals
- **DQN/PPO vs Random**: Should beat random consistently
- **DQN/PPO vs CFR**: Competitive performance (±0.5 mean return)
- **Training Stability**: Loss should decrease, reward should increase

## 🚀 Quick Commands Reference

```bash
# Evaluate your model vs CFR
python run_evaluation.py --mode head_to_head --agents dqn cfr --episodes 1000

# Full tournament with plots
python run_full_evaluation.py --agents random dqn ppo cfr --episodes 5000

# Check CFR quality
python run_evaluation.py --mode exploitability --agents cfr

# Custom evaluation with your own script
python your_evaluation_script.py
```

## 💡 Tips for Your Report

1. **Training Curves**: Show DQN/PPO learning progress
2. **Convergence**: Compare DQN/PPO convergence vs CFR's Nash convergence
3. **Head-to-Head**: Detailed analysis of each matchup
4. **Tournament**: Overall rankings with heatmap
5. **Statistics**: Use CSV exports for tables in report

## 📞 Need Help?

If you encounter issues:
1. Check the documentation files
2. Run example code in `visualization.py` (run `python visualization.py`)
3. Ask Person 3 (CFR implementation author)

## ✅ Checklist for Final Submission

- [ ] DQN model trained and saved
- [ ] PPO model trained and saved
- [ ] Evaluated vs Random (both should win)
- [ ] Evaluated vs CFR (competitive performance)
- [ ] Generated training curves
- [ ] Generated tournament visualizations
- [ ] Created summary statistics
- [ ] Saved all plots for report
- [ ] Exported data to CSV

## 🎉 You're Ready!

Everything is set up for you to:
1. ✅ Train your DQN/PPO models
2. ✅ Compare against the trained CFR agent
3. ✅ Generate publication-quality visualizations
4. ✅ Create comprehensive analysis for your report

Good luck with your implementation! 🚀
