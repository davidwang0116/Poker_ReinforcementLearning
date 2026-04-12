# CFR Agent Implementation

## Overview

This directory contains a complete implementation of **Counterfactual Regret Minimization (CFR)** for Leduc Hold'em poker. CFR is a game-theoretic algorithm that provably converges to a Nash equilibrium in two-player zero-sum games.

## What is CFR?

Unlike reinforcement learning agents (DQN, PPO) that learn from experience, CFR:
- Iteratively traverses the **full game tree**
- Accumulates **regret** for each action at each information set
- Updates strategies via **regret matching**
- The **average strategy** converges to Nash equilibrium

### Key Concepts

**Exploitability**: Measures how far a strategy is from Nash equilibrium. Lower is better:
- `0.0` = Perfect Nash equilibrium
- `< 0.1` = Excellent convergence
- `< 0.5` = Good
- `>= 1.0` = Needs more training

## Files

- `agents/cfr_agent.py` - Main CFR agent implementation
- `train_cfr.py` - Training script
- `test_cfr.py` - Unit tests
- `models/cfr/` - Saved models

## Quick Start

### 1. Train a CFR Agent

```bash
# Activate virtual environment
source .venv/bin/activate

# Quick training (1,000 iterations, ~1-2 minutes)
python train_cfr.py --iterations 1000

# Standard training (10,000 iterations, ~10-15 minutes)
python train_cfr.py --iterations 10000

# High-quality training (100,000 iterations, ~90-120 minutes)
python train_cfr.py --iterations 100000 --save-path models/cfr/cfr_100k.pkl
```

### 2. Evaluate Against Other Agents

```bash
# CFR vs Random
python run_evaluation.py --mode head_to_head --agents cfr random --episodes 1000

# Round-robin tournament (all agents)
python run_evaluation.py --mode round_robin --agents random dqn ppo cfr

# Check exploitability (Nash distance)
python run_evaluation.py --mode exploitability --agents cfr
```

### 3. Run Tests

```bash
python test_cfr.py
```

## Expected Results

### Training Convergence

With 10,000 iterations, you should see:
```
Final exploitability: ~0.01-0.02
```

With 100,000 iterations:
```
Final exploitability: ~0.001-0.005
```

### Performance vs Random Agent

CFR should consistently beat a random agent:
```
Mean return: +0.3 to +0.5
Win rate: 45-55%
```

## Implementation Details

### Architecture

The CFR agent wraps OpenSpiel's built-in CFR solver:
1. **Training**: Runs CFR iterations on the game tree
2. **Policy Extraction**: Builds a lookup table mapping observations to action probabilities
3. **Inference**: Looks up the policy for each observation and samples actions

### Key Methods

- `train(num_episodes)` - Run CFR iterations
- `step(time_step)` - Choose action using learned policy
- `save(path)` / `load(path)` - Persist/restore policy
- `get_exploitability()` - Compute Nash distance

### Limitations

1. **Scalability**: CFR requires traversing the full game tree, so it only works for small games (Leduc, Kuhn poker). For larger games, use Neural Fictitious Self-Play (NFSP).

2. **Observation Mapping**: The current implementation maps observation tensors to policies. For perfect accuracy, we would need to maintain game state synchronization.

3. **Training Time**: CFR is slower than RL per iteration because it traverses the entire game tree. However, it requires fewer iterations to converge.

## Comparison with RL Agents

| Aspect | CFR | DQN/PPO |
|--------|-----|---------|
| Learning | Game tree traversal | Experience replay |
| Convergence | Provably to Nash | No guarantees |
| Scalability | Small games only | Scales to large games |
| Training time | Slower per iteration | Faster per iteration |
| Sample efficiency | N/A (no sampling) | Requires many samples |
| Optimality | Optimal (Nash) | Approximate |

## Troubleshooting

### "Exploitability is high"
- Train for more iterations
- Typical values: 1K iters → 0.1, 10K iters → 0.01, 100K iters → 0.001

### "Agent not beating random"
- Check that the model loaded correctly
- Verify exploitability is low (< 0.1)
- Random agent can win due to variance; run more evaluation episodes

### "Training is slow"
- CFR is inherently slow for large games
- Leduc poker: ~12 iterations/second on modern hardware
- Consider using fewer iterations for testing

## References

1. Zinkevich et al. (2007) - "Regret Minimization in Games with Incomplete Information"
2. OpenSpiel documentation: https://github.com/google-deepmind/open_spiel
3. Course lecture notes: TreeSearch.pdf, DeepRL.pdf

## Future Improvements

1. **NFSP**: Implement Neural Fictitious Self-Play for scalability
2. **State Synchronization**: Maintain game state for perfect policy lookup
3. **CFR+**: Use CFR+ variant for faster convergence
4. **Visualization**: Plot exploitability over training
