# Poker RL Project — Comparing RL Algorithms in Imperfect-Information Games

## Overview

This project compares multiple reinforcement learning / game-theoretic algorithms
on **Leduc Hold'em Poker**, a standard benchmark for imperfect-information games,
using the [OpenSpiel](https://github.com/google-deepmind/open_spiel) framework.

Leduc Hold'em is a simplified poker variant with a small deck (6 cards: J, Q, K
in two suits) and two betting rounds.  Despite its simplicity, it retains the key
challenges of real poker — **hidden information, bluffing, and opponent modeling**.

#### The big idea
 
Two players compete to win **chips** (think of them as points).  Each
round, both players put some chips into a shared pile called the **pot**.
At the end of the round, whoever has the better card wins the *entire*
pot — so you want to win rounds where the pot is large, and avoid losing
rounds where you have put a lot of chips in.
 
The twist: **you can see your own card, but not your opponent's.**  This
creates all the tension.  You must decide whether to keep putting chips
in (risky if your card is weak) or quit early (safe but you lose what
you already put in).
 
---
 
#### Setup
 
- **Deck**: only 6 cards — Jack (J), Queen (Q), King (K), each in two
  suits (e.g. ♠ and ♥).  Card strength: **K > Q > J**.
- **Players**: exactly 2.
- **Starting cost ("ante")**: before anything happens, both players are
  *forced* to put 1 chip each into the pot.  This means there is always
  something at stake — you can never play for free.
 
---
 
#### How one game plays out
 
**Step 1 — Deal private cards** (automatic, no player choice)
 
The game (not the players!) randomly deals 1 card face-down to each
player.  You see your own card; you do *not* see your opponent's.
Nobody chooses their card — it is pure luck.
 
**Step 2 — Betting Round 1**
 
Players take turns.  On your turn you have up to three options:
 
| Action    | What it means | When you can do it |
|-----------|---------------|--------------------|
| **Call** (or "Check") | "I'm still in." Match whatever the opponent bet, or pay nothing if no new bet was made. | Always available. |
| **Raise** | "I'm confident — I'll put in *more* chips." Adds 2 chips to the pot and forces your opponent to respond. | Only if 0 or 1 raises have already happened this round (max 2 raises per round, then you can only Call or Fold). |
| **Fold**  | "I give up." You lose every chip you already put in the pot, but you stop the bleeding. | Always available. |
 
The round ends once both players have acted and neither wants to raise
further.
 
> **Where does the tension come from?**
> - If you have a King, you probably want to raise — but if you raise
>   too eagerly, your opponent might guess you have a King and fold
>   (so you win a small pot instead of a big one).
> - If you have a Jack, you might *bluff* by raising, hoping your
>   opponent folds a Queen — but if they call, you are likely to lose
>   a bigger pot.
> - Every action leaks information about your card.
 
**Step 3 — Deal community card** (automatic)
 
One more card is dealt face-up in the middle of the table.  **Both**
players can see it.  This card is shared — it does not belong to either
player, but it affects who wins.
 
**Step 4 — Betting Round 2**
 
Same rules as Round 1, except raises are now **4 chips** (higher
stakes in the later round).  Again, at most 2 raises allowed.
 
**Step 5 — Showdown (who wins?)**
 
If neither player folded, both reveal their private cards and compare:
 
1. **Pair beats non-pair.**  If your private card has the same rank as
   the community card (e.g. you hold Q♠ and the community card is Q♥),
   you have a "pair" and you beat any opponent who doesn't have a pair.
   (Only one player can pair with the community card, since there are
   only 2 cards of each rank in the deck.)
 
2. **Higher card wins** if neither player has a pair.  K beats Q beats J.
 
The winner takes the **entire pot** (all chips both players put in
during the whole game).  The loser gets nothing back.
 
---
 
#### Constraints at a glance
 
- Max 2 raises per betting round.
- Raise sizes are fixed: 2 chips in Round 1, 4 chips in Round 2.
- If you fold, you immediately lose everything you put in so far.
- No card trading, no card choosing — all deals are random.
 
---
 
#### A concrete example
 
```
Pot starts at 2 (1 ante from each player).
 
Private deal:  Player 0 gets Q♠,  Player 1 gets J♥.
               (Player 0 does NOT know Player 1 has J♥, and vice versa.)
 
Round 1 betting:
  Player 0: Raise  → puts 2 more chips in.   Pot = 4.
  Player 1: Call   → matches the 2 chips.     Pot = 6.
 
Community card:  K♠  (both players see it)
 
Round 2 betting:
  Player 0: Call (check, no new bet)
  Player 1: Raise → puts 4 chips in.          Pot = 10.
  Player 0: Fold  → gives up.
 
Result: Player 1 wins the pot (10 chips) even though they had the
        weakest card (J♥)!  Player 1 bluffed successfully.
        Player 0 lost the chips they had put in (1 ante + 2 raise = 3).
```
 
---
 
#### Why this game matters for RL
 
- **Hidden information** makes this fundamentally different from the
  GridWorld / CartPole problems in HW2.  An agent cannot just look at
  the "state" and compute the best move — it must reason about what
  the opponent *might* hold based on how they have been betting.
- **Bluffing** is a real, optimal strategy: even the theoretically
  perfect (Nash equilibrium) strategy sometimes raises with bad cards.
- The game is small enough (~936 distinct situations a player can be in)
  that we can compute the *exact* optimal strategy with CFR and then
  measure how close our RL agents (DQN, PPO) get to it — a luxury we
  would not have in full-scale Texas Hold'em.
 
---

### This Project

| Leduc Poker                                         |
|-----------------------------------------------------|
| Multi-agent, zero-sum game                          |
| **Imperfect information** (hidden cards)            |
| Neural-net function approximation (DQN, PPO)        |
| DQN, PPO, CFR, NFSP                                 |
| Stochastic (chance nodes for card deals)            |
| **Model-free** learning against adaptive opponents  |

---

## Team Roles

| Person | Scope | Key Files | Status |
|--------|-------|-----------|--------|
| **1 (You)** | Environment + baseline | `poker_env.py`, `agents/random_agent.py`, `evaluate.py`, `config.py` | ✓ Complete |
| **2** | Algorithm 1 + 2 | `agents/dqn_agent.py`, `agents/ppo_agent.py` | ✓ Complete |
| **3** | Algorithm 3 + analysis | `agents/cfr_agent.py`, `train_cfr.py`, `CFR_README.md` | ✓ Complete |

Everyone should understand the full pipeline end-to-end.

---

## Quick Start (Python 3.10)

```bash
# 1. Create a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows PowerShell

# 2. Install dependencies (replace with your own relative path)
pip install -r requirements.txt

# 3. Verify OpenSpiel is working
python -c "import pyspiel; print(pyspiel.registered_names())"

# 4. Train Model
#自动化训练
#项目现在采用 配置文件驱动。main.py 会根据 YAML 文件自动初始化智能体和保存路径。

# 训练基础 DQN
python main.py --config configs/dqn_basic.yaml

# 开启 PPO 自博弈训练 (Self-Play)
python main.py --config configs/ppo_selfplay.yaml

# 5. Run baseline evaluation (random vs random) (replace with your own relative path)
python run_evaluation.py

# PPO 模型 vs 随机对手 (1000局)
python run_evaluation.py --mode head_to_head --agents ppo random --episodes 1000

# 算法全家桶循环赛 (Random, DQN, PPO, CFR)
python run_evaluation.py --mode round_robin --agents random dqn ppo cfr

# CFR 专用：训练 CFR 智能体
python train_cfr.py --iterations 10000

# CFR 专用：检查 Nash 距离（可利用性）
python run_evaluation.py --mode exploitability --agents cfr
```

---

## Project Structure

```
poker_rl_project/
├── main.py                # [核心] 统一训练入口，支持 Basic 和 Self-play 模式
├── run_evaluation.py      # [核心] 评估入口，支持 1v1 对战与循环赛
├── poker_env.py           # 环境包装器，将 OpenSpiel 抽象为 TimeStep 接口
├── config.py              # 基础配置类定义
├── configs/               # [新增] 存放所有实验的 YAML 配置文件
│   ├── dqn_basic.yaml     # DQN 对阵随机对手配置
│   ├── dqn_selfplay.yaml  # DQN 自博弈配置
│   └── ppo_selfplay.yaml  # PPO 自博弈配置
├── agents/                # 算法实现目录
│   ├── dqn_agent.py       # 深度 Q 网络 (Experience Replay, Target Network)
│   ├── ppo_agent.py       # 近端策略优化 (Clipped Objective, GAE)
│   └── random_agent.py    # 随机基准
├── models/                # [新增] 训练产出的模型权重 (.pth) 存放处
│   ├── dqn/               # 自动分类存储 DQN 模型
│   └── ppo/               # 自动分类存储 PPO 模型
└── evaluate.py            # 评估逻辑库（胜率统计、回报计算）
```

> **About `results/`**: The folder is shipped empty as a convention.  Right now
> `run_evaluation.py` only prints results to the terminal.  As the project
> matures, Person 2 should save training checkpoints and loss curves here,
> and Person 3 should save comparison plots, win-rate tables, and the final
> report figures here.  It is listed in `.gitignore` so large binary files
> (model weights, etc.) don't get committed to version control.

> **About 'main.py' (训练中枢)：** 
> 通过 ConfigObject 动态解析 YAML。
>Self-play 机制：在自博弈模式下，Player 0 和 Player 1 共享同一套神经网络参数，实现“左右互搏”进化

>  **About 'poker_env.py' (环境兼容)：**
> 自动处理 Chance Nodes（发牌节点）。
> 提供 legal_actions_mask 防止智能体非法下注。

> **About 'ppo_agent.py' (Person 2)：**
>实现了优势函数估计与策略剪裁。
>注意：在评估模式（eval_mode=True）下，智能体采用确定性动作（Greedy Action）以保证表现稳定。
---

## Lecture-Note Connections

Throughout the codebase, comments of the form

```
# [LECTURE: <topic> — <slide/concept>]
```

flag where a concept from the course material appears.  Key connections:

* **MDP formulation** → `poker_env.py` wraps an imperfect-info game as a
  sequential decision process (cf. MDPs_1.pdf, slide on MDP tuple ⟨S,A,T,R,γ⟩).
* **ε-greedy exploration** → `agents/base_agent.py` helper
  (cf. ModelFreeRL.pdf, Q-learning exploration).
* **Policy gradient / PPO** → `agents/ppo_agent.py` (Person 2)
  (cf. PolicySearch.pdf, PGinPractice.pdf — softmax policies, advantage estimation, TRPO → PPO).
* **Deep Q-Networks** → `agents/dqn_agent.py` (Person 2)
  (cf. DeepRL.pdf — experience replay, target networks).
* **Minimax / game-tree search** → conceptual basis for CFR
  (cf. TreeSearch.pdf, DeepRL.pdf slide "Extension to 2 player games").
* **Evaluation via exploitability** → `evaluate.py`
  (cf. DeepRL.pdf — Nash equilibrium in zero-sum games).

---

## Notes for Teammates

* **Person 2**: Subclass `BaseAgent` in `agents/base_agent.py`.  Your agents
  must implement `step()`, `train()`, and `save()`/`load()`.  The evaluation
  harness in `evaluate.py` already handles everything else.
* **Person 3**: For CFR / NFSP you can either wrap OpenSpiel's built-in
  solvers or implement from scratch.  `evaluate.py` already computes
  exploitability via OpenSpiel's `exploitability` module — just call
  `evaluate_exploitability()`.
