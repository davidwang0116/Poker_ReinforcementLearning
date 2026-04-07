"""
dqn_agent.py — Deep Q-Network agent for Leduc Hold'em.

*** PERSON 2: Implement this file. ***

────────────────────────────────────────────────────────────
Lecture-note connections
────────────────────────────────────────────────────────────

[LECTURE: DeepRL — DQN approximates Q(s, a) with a neural
 network Q(s, a; θ).  Two key innovations stabilise training:

 1. **Experience replay**: store (s, a, r, s') transitions in
    a buffer and sample mini-batches uniformly.  This breaks
    temporal correlations that cause instability.

 2. **Target network**: maintain a *frozen* copy θ⁻ of the
    network weights, updated every C steps.  The TD target
      y = r + γ max_a' Q(s', a'; θ⁻)
    is computed with θ⁻, preventing the "moving target" problem.]

[LECTURE: ModelFreeRL — Q-learning update:
     Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
 DQN replaces the tabular Q with a neural net and uses
 gradient descent on the squared TD error:
     L(θ) = E[(y - Q(s, a; θ))²]
 This is a direct extension of the Q-learning from HW2
 Section 4 to non-linear function approximation.]

[LECTURE: SarsaAndLambda — DQN is *off-policy* (like Q-learning,
 unlike SARSA) because the target uses max_a' rather than
 the action actually taken.  This means we can learn from
 experience gathered by older policies stored in the replay buffer.]

────────────────────────────────────────────────────────────
Implementation guide
────────────────────────────────────────────────────────────

1. Subclass `BaseAgent`.
2. Build a simple MLP: obs_size → hidden → hidden → num_actions.
3. Implement a replay buffer (list or deque of transitions).
4. In `step()`:
   - Convert obs to tensor, forward through Q-network.
   - Mask illegal actions to -∞.
   - Use ε-greedy (see `self._epsilon_greedy` in BaseAgent).
   - Store transition in replay buffer.
5. In `train()`:
   - Sample mini-batch from buffer.
   - Compute TD targets with target network.
   - MSE loss, backprop, optimizer step.
   - Every C steps, sync target network.
6. Implement `save()` / `load()` with `torch.save` / `torch.load`.
"""

from __future__ import annotations
"""
dqn_agent.py — Deep Q-Network agent for Leduc Hold'em with GPU support.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Optional

from agents.base_agent import BaseAgent
from config import DQNConfig, SEED

class QNetwork(nn.Module):
    """一个简单的多层感知机 (MLP)，用于近似 Q(s, a)。"""
    def __init__(self, obs_size: int, num_actions: int, hidden_sizes: List[int]):
        super(QNetwork, self).__init__()
        layers = []
        in_size = obs_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, num_actions))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DQNAgent(BaseAgent):
    def __init__(self, player_id: int, num_actions: int, obs_size: int, config: DQNConfig):
        super().__init__(player_id, num_actions, rng_seed=SEED + player_id)
        self.config = config
        self.obs_size = obs_size
        self.eval_mode = False

        # 🟢 GPU 自动检测与配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent {player_id} is using device: {self.device}")
        
        # 1. 初始化网络并推送到设备 (GPU/CPU)
        self.q_net = QNetwork(obs_size, num_actions, config.hidden_sizes).to(self.device)
        self.target_net = QNetwork(obs_size, num_actions, config.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        
        # 2. 经验回放缓冲区
        self.memory = deque(maxlen=config.replay_buffer_size)
        
        self.train_steps = 0
        self.last_obs = None
        self.last_action = None
        self.epsilon = config.epsilon_start
        self.last_loss = 0.0 # 用于进度条打印

    def step(self, state) -> int:
        """选择动作，完美兼容 pyspiel.State, TimeStep 以及 numpy 数组。"""
        
        # 1. 提取观察向量 (Observation / Information State)
        if hasattr(state, "obs"):
            # 兼容 PokerEnv.TimeStep
            obs_vec = state.obs
        elif hasattr(state, "information_state_tensor"):
            # 兼容 pyspiel.State (Training mode)
            obs_vec = state.information_state_tensor(self.player_id)
        elif isinstance(state, (np.ndarray, list)):
            # 直接传入向量的情况
            obs_vec = state
        else:
            raise AttributeError(f"无法从 {type(state)} 中提取观察向量")

        obs_tensor = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device)

        # 2. 提取合法动作 (解决 TypeError: 'list' object is not callable)
        if hasattr(state, "legal_actions"):
            la = state.legal_actions
            # 如果是方法则调用，如果是列表则直接使用
            legal_actions = la() if callable(la) else la
        elif hasattr(state, "legal_actions_mask"):
            # 从掩码中还原动作列表
            legal_actions = [i for i, m in enumerate(state.legal_actions_mask) if m > 0]
        else:
            raise AttributeError(f"无法从 {type(state)} 获取合法动作列表")

        # 3. 经验回放存储 (仅在训练模式下)
        if not self.eval_mode and self.last_obs is not None:
            # 自动获取奖励
            reward = state.reward if hasattr(state, "reward") else 0.0
            done = state.done if hasattr(state, "done") else False
            self.memory.append((self.last_obs, self.last_action, reward, obs_vec, done))

        # 4. 网络推理
        with torch.no_grad():
            q_values = self.q_net(obs_tensor).cpu().numpy()
        
        # 5. 执行策略
        action = self._epsilon_greedy(
            q_values, 
            legal_actions, 
            self.epsilon if not self.eval_mode else 0.0
        )

        if not self.eval_mode:
            self.last_obs = obs_vec
            self.last_action = action
            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

        return action

    def on_episode_end(self, final_reward: float):
        if not self.eval_mode and self.last_obs is not None:
            self.memory.append((
                self.last_obs, 
                self.last_action, 
                final_reward, 
                np.zeros(self.obs_size), 
                True
            ))
        self.last_obs = None
        self.last_action = None

    def train(self) -> float:
        """从缓冲区采样并执行一个梯度更新步骤，返回当前 Loss。"""
        if len(self.memory) < self.config.batch_size:
            return 0.0

        # 采样 Batch
        batch = random.sample(self.memory, self.config.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        # 🟢 关键：将所有训练数据推送到 GPU
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # 当前 Q 值: Q(s, a; θ)
        current_q = self.q_net(obs).gather(1, actions).squeeze(1)

        # 计算 TD 目标: y = r + γ * max_a' Q(s', a'; θ⁻)
        with torch.no_grad():
            next_q = self.target_net(next_obs).max(1)[0]
            target_q = rewards + (1 - dones) * self.config.gamma * next_q

        # MSE 损失计算
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        self.last_loss = loss.item()
        
        # 每隔 target_update_freq 步同步目标网络
        if self.train_steps % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        return self.last_loss

    def save(self, path: str):
        # 确保保存路径的文件夹存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_net': self.q_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps
        }, path)

    def load(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.q_net.load_state_dict(checkpoint['q_net'])
            self.target_net.load_state_dict(checkpoint['q_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.train_steps = checkpoint.get('train_steps', 0)
