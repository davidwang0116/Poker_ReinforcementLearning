"""
ppo_agent.py — Proximal Policy Optimization agent for Leduc Hold'em.

*** PERSON 2: Implement this file. ***

────────────────────────────────────────────────────────────
Lecture-note connections
────────────────────────────────────────────────────────────

[LECTURE: PolicySearch — Policy gradient theorem:
     ∇J(θ) = E_π [Σ_k ∇ log π(a_k|s_k; θ) · γ^(k-1) · A(s_k, a_k)]
 where A(s, a) = Q(s, a) - V(s) is the *advantage function*.
 Using the advantage rather than raw returns dramatically
 reduces variance (cf. PolicySearch.pdf, eq. 11.44).]

[LECTURE: PGinPractice — PPO clips the surrogate objective:
     L^CLIP(θ) = E[ min(r_t(θ) Â_t,  clip(r_t(θ), 1-ε, 1+ε) Â_t) ]
 where r_t(θ) = π_θ(a|s) / π_θ_old(a|s) is the importance
 sampling ratio.  Clipping prevents destructively large
 updates, achieving similar stability to TRPO without the
 expensive second-order optimisation.

 PPO also adds:
   - An entropy bonus  (encourages exploration)
   - A value-function loss  (trains the critic)]

[LECTURE: PGinPractice — PPO is currently the most popular
 deep RL algorithm (as of Winter 2026).  It combines value
 function approximation with a differentiable policy,
 unlike DQN which derives the policy implicitly from Q-values.]

────────────────────────────────────────────────────────────
Implementation guide
────────────────────────────────────────────────────────────

1. Subclass `BaseAgent`.
2. Build an actor-critic MLP:
   - Shared trunk: obs_size → hidden → hidden
   - Policy head:  hidden → num_actions  (logits)
   - Value head:   hidden → 1
3. In `step()`:
   - Forward pass → logits + value.
   - Mask illegal actions, softmax → sample action.
   - Store (obs, action, log_prob, value, reward, done).
4. In `train()`:
   - Collect rollouts of length `rollout_length`.
   - Compute advantages with GAE(λ).
   - For `epochs_per_update` epochs:
       - Compute clipped surrogate loss + value loss + entropy.
       - Backprop, optimizer step.
5. Implement `save()` / `load()`.
"""

from __future__ import annotations

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from agents.base_agent import BaseAgent
# from config import PPOConfig


"""
ppo_agent.py — Proximal Policy Optimization agent for Leduc Hold'em.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os

class PPOPolicy(nn.Module):
    def __init__(self, obs_size, num_actions, hidden_sizes=[256, 256]):
        super().__init__()
        # 共享骨干网络 (Shared Trunk)
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU()
        )
        # Actor: 输出动作的 Logits
        self.actor_head = nn.Linear(hidden_sizes[1], num_actions)
        # Critic: 输出状态价值 V(s)
        self.critic_head = nn.Linear(hidden_sizes[1], 1)

    def forward(self, obs):
        latent = self.trunk(obs)
        logits = self.actor_head(latent)
        value = self.critic_head(latent)
        return logits, value

class PPOAgent:
    def __init__(self, player_id, num_actions, obs_size, config):
        self.player_id = player_id
        self.num_actions = num_actions
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 从配置中读取参数，若无则使用默认值
        hidden_sizes = getattr(config, "hidden_sizes", [256, 256])
        self.policy = PPOPolicy(obs_size, num_actions, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        
        # 轨迹缓存
        self.states, self.actions, self.log_probs, self.values, self.rewards = [], [], [], [], []

    def step(self, ts, eval_mode=False):
        """
        完美适配 PokerEnv.TimeStep (NamedTuple)
        """
        # 1. 自动处理两种输入来源：
        #    - 训练模式 (main.py): 传入的是 pyspiel.State
        #    - 评估模式 (run_evaluation.py): 传入的是 poker_env.TimeStep
        
        if hasattr(ts, 'information_state_tensor'):
            # 🟢 训练模式：直接从 State 获取
            obs = ts.information_state_tensor(self.player_id)
            legal_actions = ts.legal_actions()
        else:
            # 🔵 评估模式：从 PokerEnv.TimeStep 获取
            # 注意：根据你的源码，字段名是 .obs 而不是 .info_state
            obs = ts.obs 
            legal_actions = ts.legal_actions
        
        # 2. 转换为 Tensor 推理
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            logits, value = self.policy(obs_t)
        
        # 3. 动作掩码 (防止非法动作)
        mask = torch.full((self.num_actions,), -1e10, device=self.device)
        mask[legal_actions] = 0.0
        masked_logits = logits.squeeze(0) + mask

        if eval_mode:
            # 评估时：确定性动作 (Greedy)
            action = torch.argmax(masked_logits).item()
        else:
            # 训练时：概率采样 (Sampling)
            dist = torch.distributions.Categorical(logits=masked_logits)
            action_t = dist.sample()
            action = action_t.item()
            
            # 记录轨迹用于训练
            self.states.append(obs)
            self.actions.append(action)
            self.log_probs.append(dist.log_prob(action_t).item())
            self.values.append(value.item())
            
        return action

    def update(self, final_reward):
        """每局结束调用：将终局奖励分配给本局所有动作并触发训练"""
        # 扑克奖励稀疏，整局所有 step 共享同一个 final_reward
        num_steps = len(self.states) - len(self.rewards)
        self.rewards.extend([final_reward] * num_steps)
        
        # 当积累了一定规模的样本（如 512 步）后进行模型更新
        if len(self.states) >= 512:
            self._train()

    def _train(self):
        # 转换为 Tensor
        s = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        a = torch.tensor(self.actions, device=self.device)
        old_log_p = torch.tensor(self.log_probs, device=self.device)
        old_v = torch.tensor(self.values, device=self.device)
        returns = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        
        # 计算优势函数 A(s,a) = Return - V(s)
        advantages = returns - old_v

        for _ in range(self.config.ppo_epochs):
            logits, values = self.policy(s)
            dist = Categorical(logits=logits)
            new_log_p = dist.log_prob(a)
            entropy = dist.entropy().mean()
            
            # PPO 核心：计算 Ratio 并 Clipping
            ratio = torch.exp(new_log_p - old_log_p)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # 总损失：策略损失 + 价值损失 - 熵加成（鼓励探索）
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5) # 梯度裁剪
            self.optimizer.step()
            
        # 清空缓冲区
        self.states, self.actions, self.log_probs, self.values, self.rewards = [], [], [], [], []

    def save(self, path):
        """保存策略网络权重"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 只需要保存 policy 的 state_dict
        torch.save(self.policy.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        """加载策略网络权重"""
        if not os.path.exists(path):
            print(f"Warning: {path} 不存在，加载失败。")
            return
        
        # 确保加载到正确的设备 (CPU/CUDA)
        state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        # 🟢 关键：切换到评估模式，关闭 Dropout 等（如果有的话）
        self.policy.eval() 
        print(f"Model successfully loaded from {path}")
    def on_episode_end(self, reward):
        """
        兼容性方法。
        在评估脚本中，每局结束会调用此函数。
        对于 PPO 评估，我们不需要在这里做任何操作。
        """
        pass