import os
import torch
import numpy as np
import pyspiel
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from agents.dqn_agent import DQNAgent
from visualization import plot_training_curves

class ConfigObject:
    def __init__(self, d):
        for k, v in d.items():
            # 强制转换数值类型，防止 YAML 解析为字符串导致运算错误
            if k in ['target_update_freq', 'batch_size', 'num_episodes', 'replay_buffer_size', 'seed', 'eval_every']:
                v = int(v)
            elif k in ['lr', 'gamma', 'epsilon_decay', 'epsilon_start', 'epsilon_end']:
                v = float(v)
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
                agent.on_episode_end(returns[pid])
                agent.train()
    return returns

def evaluate_vs_random(game, trained_agent, num_episodes=1000):
    class RandomAgent:
        def step(self, s): return np.random.choice(s.legal_actions())
    
    original_eps = trained_agent.epsilon
    trained_agent.epsilon = 0.0 
    
    eval_agents = {0: trained_agent, 1: RandomAgent()}
    wins = 0
    for _ in range(num_episodes):
        returns = run_episode(game, eval_agents, is_training=False)
        if returns[0] > 0: wins += 1
            
    trained_agent.epsilon = original_eps 
    return wins / num_episodes

def train_and_log(cfg):
    # --- 1. 构建唯一的实验标识符 (ID) ---
    params_id = (
        f"ep{cfg.training.num_episodes}_lr{cfg.agent.lr}_s{cfg.seed}_"
        f"g{cfg.agent.gamma}_b{cfg.agent.batch_size}_ed{cfg.agent.epsilon_decay}_"
        f"tuf{cfg.agent.target_update_freq}_rbs{cfg.agent.replay_buffer_size}_e{cfg.agent.epsilon_end}"
    )

    # 路径处理
    base_save_dir = os.path.dirname(os.path.abspath(cfg.training.save_path))
    os.makedirs(base_save_dir, exist_ok=True)
    
    # 动态生成的模型保存路径
    dynamic_model_path = os.path.join(base_save_dir, f"dqn_sp_{params_id}.pth")
    
    game = pyspiel.load_game(cfg.game_name)
    num_actions = game.num_distinct_actions()
    obs_size = game.information_state_tensor_size()
    
    # 2. 初始化 Agents
    a0 = DQNAgent(0, num_actions, obs_size, cfg.agent)
    a1 = DQNAgent(1, num_actions, obs_size, cfg.agent)
    a1.q_net = a0.q_net 
    agents = {0: a0, 1: a1}

    # 3. 训练循环
    history = {"loss": [], "reward": [], "epsilon": []}
    pbar = tqdm(range(cfg.training.num_episodes), desc="DQN Self-Play")
    
    for episode in pbar:
        returns = run_episode(game, agents)
        history["reward"].append(returns[0])
        if hasattr(a0, 'last_loss') and a0.last_loss:
            history["loss"].append(a0.last_loss)
        history["epsilon"].append(a0.epsilon)
            
        if episode % 1000 == 0:
            pbar.set_postfix({"Eps": f"{a0.epsilon:.2f}"})

        # 定期保存到动态路径
        if episode > 0 and episode % cfg.training.eval_every == 0:
            a0.save(dynamic_model_path)

    # 4. 自动评估
    print(f"\n[Evaluation] 对阵随机对手进行 1000 局测试...")
    win_rate = evaluate_vs_random(game, a0, num_episodes=1000)
    print(f"胜率: {win_rate:.2%}")

    # 5. 记录到 Excel (增加 Model_Name 字段实现一一对应)
    log_file = "experiment_logs.xlsx"
    log_data = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Win_Rate": win_rate,
        "Episodes": cfg.training.num_episodes,
        "LR": cfg.agent.lr,
        "Seed": cfg.seed,
        "Gamma": cfg.agent.gamma,
        "BatchSize": cfg.agent.batch_size,
        "Eps_Decay": cfg.agent.epsilon_decay,
        "Target_Update_Freq": cfg.agent.target_update_freq,
        "Replay_Buffer_Size": cfg.agent.replay_buffer_size,
        "Epsilon_End": cfg.agent.epsilon_end,
        "Model_File": f"dqn_sp_{params_id}.pth" # 关联点
    }
    
    df_new = pd.DataFrame([log_data])
    if os.path.exists(log_file):
        try:
            df_old = pd.read_excel(log_file)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        except:
            df_final = df_new
    else:
        df_final = df_new
    df_final.to_excel(log_file, index=False)
    print(f"✓ 实验数据已记入 {log_file}")

    # 6. 最终模型保存 (使用动态路径)
    a0.save(dynamic_model_path)
    print(f"✓ 模型已保存至: {dynamic_model_path}")

    # 7. 可视化 (图片名与模型名完全一致)
    plot_dir = os.path.join(base_save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fig_path = os.path.join(plot_dir, f"dqn_sp_{params_id}.png")
    
    plot_metrics = {k: v for k, v in history.items() if len(v) > 0}
    plot_training_curves(
        plot_metrics, 
        title=f"DQN Self-Play (LR:{cfg.agent.lr} Seed:{cfg.seed})",
        save_path=fig_path, 
        show=False
    )
    print(f"✓ 收敛图已保存至: {fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dqn_selfplay.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
        cfg = ConfigObject(cfg_dict)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    train_and_log(cfg)