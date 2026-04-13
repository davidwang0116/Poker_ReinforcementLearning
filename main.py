import os
import torch
import numpy as np
import pyspiel
import yaml
import argparse
from tqdm import tqdm
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from visualization import plot_training_curves  # 导入绘图工具

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
                agent.update(returns[pid])
    return returns

def train(cfg):
    save_path = os.path.abspath(cfg.training.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # === 新增：初始化历史记录字典，用于可视化 ===
    history = {
        "loss": [],
        "reward": [],
        "epsilon": []
    }
    
    game = pyspiel.load_game(cfg.game_name)
    num_actions = game.num_distinct_actions()
    obs_size = game.information_state_tensor_size()
    
    def create_agent(pid):
        algo = getattr(cfg, "algo", "dqn")
        if algo == "ppo":
            return PPOAgent(pid, num_actions, obs_size, cfg.agent)
        return DQNAgent(pid, num_actions, obs_size, cfg.agent)

    a0 = create_agent(0)
    if cfg.mode == "self_play":
        a1 = create_agent(1)
        if hasattr(a0, 'q_net'): a1.q_net = a0.q_net
        agents = {0: a0, 1: a1}
    else:
        class Rand:
            def step(self, s): return np.random.choice(s.legal_actions())
        agents = {0: a0, 1: Rand()}

    algo_name = getattr(cfg, "algo", "dqn").upper()
    pbar = tqdm(range(cfg.training.num_episodes), desc=f"Training {algo_name}")
    
    for episode in pbar:
        # 获取本局回报
        returns = run_episode(game, agents)
        
        # === 新增：收集指标数据 ===
        # 记录玩家0（训练的Agent）的回报
        history["reward"].append(returns[0])
        
        # 记录 Loss (从 DQNAgent 的 last_loss 属性中提取)
        if hasattr(a0, 'last_loss') and a0.last_loss is not None:
            history["loss"].append(a0.last_loss)
            
        # 记录 Epsilon 衰减情况
        if hasattr(a0, 'epsilon'):
            history["epsilon"].append(a0.epsilon)
        
        if episode % 100 == 0:
            status = {"Mode": cfg.mode}
            if hasattr(a0, 'epsilon'): status["Eps"] = f"{a0.epsilon:.2f}"
            pbar.set_postfix(status)

        if episode > 0 and episode % cfg.training.eval_every == 0:
            a0.save(save_path)

    # 保存模型
    a0.save(save_path)
    print(f"Success! Model saved to {save_path}")

    # === 新增：训练结束后生成并保存图表 ===
    plot_dir = os.path.join(os.path.dirname(save_path), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    fig_path = os.path.join(plot_dir, f"{algo_name.lower()}_training_curves.png")
    
    # 过滤掉空的指标（例如 PPO 可能没有 epsilon）
    plot_metrics = {k: v for k, v in history.items() if len(v) > 0}
    
    plot_training_curves(
        plot_metrics,
        title=f"{algo_name} Training Progress",
        save_path=fig_path,
        show=False # 设为 False 以便在服务器环境下静默保存
    )
    print(f"✓ Training plots saved to {fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = ConfigObject(yaml.safe_load(f))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    train(cfg)