"""
run_evaluation.py — Main entry-point for running experiments.

Usage
-----
    # Run random-vs-random baseline (sanity check)
    python run_evaluation.py

    # Person 2: after implementing DQN
    python run_evaluation.py --agents random dqn

    # Full round-robin after all agents are ready
    python run_evaluation.py --mode round_robin --agents random dqn ppo cfr

    # Exploitability check (tabular policies only, e.g. CFR)
    python run_evaluation.py --mode exploitability --agents cfr
"""

from __future__ import annotations

import argparse
import sys
import os
import yaml
import numpy as np

from poker_env import PokerEnv
from agents.random_agent import RandomAgent
from evaluate import (
    evaluate_head_to_head,
    evaluate_round_robin,
    print_round_robin,
)
from config import SEED, NUM_EVAL_EPISODES, DQNConfig

class ConfigObject:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, ConfigObject(v) if isinstance(v, dict) else v)


# ── Agent factory ───────────────────────────────────────

def make_agent(name: str, player_id: int, env: PokerEnv, **kwargs):
    """Instantiate an agent by name.

    Extend this function when new agents are added.
    """
    name = name.lower()
    if name == "random":
        return RandomAgent(
            player_id=player_id,
            num_actions=env.num_actions,
            rng_seed=SEED + player_id,
        )
    # ── Person 2: DQN ──
    elif name == "dqn":
        from agents.dqn_agent import DQNAgent
        # 实例化时传入对应的配置
        agent = DQNAgent(player_id=player_id, num_actions=env.num_actions,
                         obs_size=env.obs_size, config=DQNConfig())
        
        # 加载训练好的模型权重
        # 这里默认加载训练结束后的模型，或者你可以通过参数传入具体路径
        load_path = kwargs.get("load_path", "models/dqn/dqn_selfplay_100w_final.pth")
        if os.path.exists(load_path):
            print(f"Loading DQN model from {load_path}...")
            agent.load(load_path)
            agent.eval_mode = True  # 评估时关闭探索
        else:
            print(f"Warning: No model found at {load_path}, using untrained agent.")
            
        return agent

    # ── Person 2: PPO ──
    elif name == "ppo":
        from agents.ppo_agent import PPOAgent
        
        # 🟢 获取 PPO 配置：尝试从 YAML 加载，否则使用默认
        ppo_config_path = "configs/ppo_selfplay.yaml"
        if os.path.exists(ppo_config_path):
            with open(ppo_config_path, 'r') as f:
                cfg_data = yaml.safe_load(f)
                ppo_cfg = ConfigObject(cfg_data['agent'])
        else:
            # 备选方案：手动创建一个包含必要属性的对象
            class DefaultPPOConfig:
                lr = 0.0003
                hidden_sizes = [256, 256]
            ppo_cfg = DefaultPPOConfig()

        agent = PPOAgent(player_id=player_id, num_actions=env.num_actions,
                         obs_size=env.obs_size, config=ppo_cfg)
        
        load_path = kwargs.get("load_path") or "models/ppo/ppo_selfplay_final.pth"
        if os.path.exists(load_path):
            print(f"Loading PPO model from {load_path}...")
            agent.load(load_path)
            # 评估模式：PPO 在 step 中通常有 eval_mode 参数来选择 argmax
        else:
            print(f"Warning: No PPO model at {load_path}")
        return agent

    else:
        raise ValueError(f"Unknown agent: {name!r}")

    # ── Person 3: CFR / NFSP ──
    # elif name == "cfr":
    #     from agents.cfr_agent import CFRAgent
    #     agent = CFRAgent(player_id=player_id, num_actions=env.num_actions,
    #                      game=env.get_game(), config=CFRConfig())
    #     if kwargs.get("load_path"):
    #         agent.load(kwargs["load_path"])
    #     return agent


# ── CLI ─────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Poker RL evaluation harness")
    p.add_argument(
        "--mode",
        choices=["head_to_head", "round_robin", "exploitability"],
        default="head_to_head",
        help="Evaluation mode (default: head_to_head).",
    )
    p.add_argument(
        "--agents",
        nargs="+",
        default=["random", "random"],
        help="Agent names.  For head_to_head, supply exactly 2.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=NUM_EVAL_EPISODES,
        help=f"Number of evaluation episodes (default: {NUM_EVAL_EPISODES}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED}).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    env = PokerEnv(seed=args.seed)
    print(f"Game:        {env.game}")
    print(f"Players:     {env.num_players}")
    print(f"Actions:     {env.num_actions}")
    print(f"Obs size:    {env.obs_size}")
    print(f"Eval mode:   {args.mode}")
    print(f"Episodes:    {args.episodes}")
    print()

    if args.mode == "head_to_head":
        if len(args.agents) != 2:
            print("ERROR: head_to_head mode requires exactly 2 agents.",
                  file=sys.stderr)
            sys.exit(1)
        a = make_agent(args.agents[0], player_id=0, env=env)
        b = make_agent(args.agents[1], player_id=1, env=env)
        result = evaluate_head_to_head(
            a, b,
            num_episodes=args.episodes,
            env=env,
            agent_a_name=args.agents[0],
            agent_b_name=args.agents[1],
        )
        print(result.summary())

    elif args.mode == "round_robin":
        agents_dict = {}
        for name in args.agents:
            agents_dict[name] = make_agent(name, player_id=0, env=env)
        rr = evaluate_round_robin(agents_dict, num_episodes=args.episodes)
        print_round_robin(rr)

    elif args.mode == "exploitability":
        # Only works for agents that expose a tabular policy
        # (e.g., CFR).  Stub for Person 3.
        print("Exploitability evaluation not yet implemented.")
        print("Person 3: implement this after CFR agent is ready.")
        sys.exit(0)


if __name__ == "__main__":
    main()
