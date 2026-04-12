# How to Use Visualization Tools - Quick Guide for Person 2

## 🎯 Goal

Add visualization to your DQN/PPO training and compare against the trained CFR model.

## ⚡ Super Quick Start (30 seconds)

```bash
# After training your models, just run this:
python run_full_evaluation.py --agents random dqn ppo cfr --episodes 1000

# Done! Check results/YYYYMMDD_HHMMSS/ for all plots
```

## 📊 Option 1: Automated (Recommended)

### One Command Does Everything

```bash
python run_full_evaluation.py --agents random dqn ppo cfr --episodes 5000
```

**Generates:**
- ✅ Tournament heatmap (who beats who)
- ✅ Agent rankings bar chart
- ✅ Head-to-head detailed comparisons
- ✅ Summary statistics (CSV)
- ✅ All data exported

**Output location**: `results/YYYYMMDD_HHMMSS/`

## 🔧 Option 2: Add to Your Training Code

### For DQN Training

Add these 3 lines to your training script:

```python
# At the top
from visualization import plot_training_curves

# In your train() method, collect metrics
loss_history = []
reward_history = []

for episode in range(num_episodes):
    # ... your training code ...
    loss_history.append(loss)
    reward_history.append(episode_reward)

# After training, add this:
plot_training_curves(
    {'loss': loss_history, 'reward': reward_history},
    title='DQN Training',
    save_path='results/dqn_training.png'
)
```

### For PPO Training

```python
# At the top
from visualization import plot_training_curves

# Collect PPO metrics
policy_loss_history = []
value_loss_history = []
reward_history = []

for episode in range(num_episodes):
    # ... your training code ...
    policy_loss_history.append(policy_loss)
    value_loss_history.append(value_loss)
    reward_history.append(episode_reward)

# After training, visualize
plot_training_curves(
    {
        'policy_loss': policy_loss_history,
        'value_loss': value_loss_history,
        'reward': reward_history
    },
    title='PPO Training',
    save_path='results/ppo_training.png'
)
```

## 🏆 Competition Commands

### Test Your Models vs CFR

```bash
# DQN vs CFR
python run_evaluation.py --mode head_to_head --agents dqn cfr --episodes 1000

# PPO vs CFR
python run_evaluation.py --mode head_to_head --agents ppo cfr --episodes 1000

# Full tournament
python run_evaluation.py --mode round_robin --agents random dqn ppo cfr
```

## 📈 Available Functions

### 1. Training Curves
```python
from visualization import plot_training_curves
plot_training_curves(metrics_dict, save_path='results/training.png')
```

### 2. Head-to-Head Comparison
```python
from visualization import plot_head_to_head
plot_head_to_head('DQN', 'CFR', dqn_returns, cfr_returns, 
                 save_path='results/dqn_vs_cfr.png')
```

### 3. Tournament Heatmap
```python
from visualization import plot_round_robin_heatmap
plot_round_robin_heatmap(payoff_matrix, agent_names,
                        save_path='results/heatmap.png')
```

### 4. Agent Rankings
```python
from visualization import plot_round_robin_ranking
plot_round_robin_ranking(payoff_matrix, agent_names,
                        save_path='results/rankings.png')
```

## 📝 For Your Report

### Minimum Required Plots

1. **Training curves** for DQN and PPO
2. **Tournament heatmap** (all agents)
3. **Agent rankings** bar chart

Generate all three with:
```bash
python run_full_evaluation.py --agents random dqn ppo cfr --episodes 5000
```

### Optional (But Impressive) Plots

4. **DQN vs CFR** detailed comparison
5. **PPO vs CFR** detailed comparison
6. **Training comparison** (DQN vs PPO learning curves)

## 🎨 Example Output

After running `run_full_evaluation.py`, you get:

```
results/20260412_180000/
├── tournament_heatmap.png      # Figure 1 for report
├── tournament_ranking.png      # Figure 2 for report
├── h2h_DQN_vs_CFR.png         # Figure 3 for report
├── h2h_PPO_vs_CFR.png         # Figure 4 for report
├── tournament_summary.csv      # Table 1 for report
└── config.txt                 # Experiment settings
```

## ⏱️ Time Estimates

| Task | Time | Command |
|------|------|---------|
| Pull CFR code | 1 min | `git checkout feature/cfr-implementation` |
| Install dependencies | 2 min | `pip install -r requirements.txt` |
| Test CFR model | 1 min | `python run_evaluation.py --mode head_to_head --agents cfr random --episodes 100` |
| Full evaluation | 5-10 min | `python run_full_evaluation.py --agents random dqn ppo cfr --episodes 1000` |
| **Total** | **~10 min** | **Ready to use!** |

## ✅ Checklist

- [ ] Pulled feature/cfr-implementation branch
- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Tested CFR model works
- [ ] Trained your DQN model
- [ ] Trained your PPO model
- [ ] Ran full evaluation
- [ ] Generated all plots
- [ ] Saved results for report

## 🚀 Ready to Start?

1. **Read**: `PARTNER_GUIDE.md` (comprehensive guide)
2. **Run**: `python run_full_evaluation.py --agents random dqn ppo cfr --episodes 1000`
3. **Done**: Use plots from `results/YYYYMMDD_HHMMSS/`

That's it! The visualization system is ready to use. 🎉

## 📚 More Resources

- **Detailed examples**: `VISUALIZATION_INTEGRATION_EXAMPLES.md`
- **Full API docs**: `VISUALIZATION_GUIDE.md`
- **CFR info**: `CFR_README.md`

## 💡 Pro Tip

The easiest workflow:
```bash
# 1. Train your models (your existing code)
python main.py --config configs/dqn_selfplay.yaml
python main.py --config configs/ppo_selfplay.yaml

# 2. Generate all visualizations (one command)
python run_full_evaluation.py --agents random dqn ppo cfr --episodes 5000

# 3. Done! All plots ready for report
```

No code changes needed! 🎯
