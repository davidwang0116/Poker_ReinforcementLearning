import os
import torch
import numpy as np
import pyspiel
import yaml
import argparse
from tqdm import tqdm
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent

class ConfigObject:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, ConfigObject(v) if isinstance(v, dict) else v)

def run_episode(game, agents, is_training=True):
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            state.apply_action(np.random.choice(outcomes, p=probs))
        else:
            cur_player = state.current_player()
            action = agents[cur_player].step(state)
            state.apply_action(action)
    
    returns = state.returns()
    if is_training:
        for pid, agent in agents.items():
            if isinstance(agent, DQNAgent):
                agent.on_episode_end(returns[pid]); agent.train()
            elif isinstance(agent, PPOAgent):
                agent.update(returns[pid]) # PPO 传入本局回报
    return returns

def train(cfg):
    save_path = os.path.abspath(cfg.training.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    game = pyspiel.load_game(cfg.game_name)
    num_actions = game.num_distinct_actions()
    obs_size = game.information_state_tensor_size()
    
    # 1. 初始化 Agents (支持多算法切换)
    def create_agent(pid):
        if getattr(cfg, "algo", "dqn") == "ppo":
            return PPOAgent(pid, num_actions, obs_size, cfg.agent)
        return DQNAgent(pid, num_actions, obs_size, cfg.agent)

    a0 = create_agent(0)
    if cfg.mode == "self_play":
        a1 = create_agent(1)
        # 自博弈：共享网络参数
        if hasattr(a0, 'policy'): a1.policy = a0.policy
        if hasattr(a0, 'q_net'): a1.q_net = a0.q_net
        agents = {0: a0, 1: a1}
    else:
        class Rand:
            def step(self, s): return np.random.choice(s.legal_actions())
        agents = {0: a0, 1: Rand()}

    # 2. 训练循环
    pbar = tqdm(range(cfg.training.num_episodes), desc=f"Training {cfg.algo.upper()}")
    for episode in pbar:
        run_episode(game, agents)
        
        if episode % 100 == 0:
            status = {"Mode": cfg.mode}
            if hasattr(a0, 'epsilon'): status["Eps"] = f"{a0.epsilon:.2f}"
            pbar.set_postfix(status)

        if episode > 0 and episode % cfg.training.eval_every == 0:
            a0.save(save_path)

    a0.save(save_path)
    print(f"Success! Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = ConfigObject(yaml.safe_load(f))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    train(cfg)
