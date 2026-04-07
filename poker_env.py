"""
poker_env.py — OpenSpiel environment wrapper.

Provides a clean, consistent API that all agents (random, DQN, PPO,
CFR/NFSP) interact with.  The wrapper handles:

  1. Creating the OpenSpiel game and initial state.
  2. Advancing through *chance nodes* (card deals) automatically so
     that agents only ever see *decision* nodes.
  3. Converting OpenSpiel information-state tensors to numpy arrays
     suitable for neural-network input.
  4. Exposing action masks so agents know which actions are legal.

────────────────────────────────────────────────────────────
Lecture-note connections
────────────────────────────────────────────────────────────

[LECTURE: MDPs_1 — An MDP is ⟨S, A, T, R, γ⟩.  In an
 *extensive-form game* the state is only partially observable;
 each player sees an "information state" that groups together
 all game histories indistinguishable to that player.  This is
 analogous to POMDPs but with multiple agents.]

[LECTURE: ModelFreeRL — The RL loop:
   observe state → choose action → receive reward → observe next state
 is preserved here, except we must also handle *chance* nodes
 (the environment's own stochastic "actions", e.g. dealing cards)
 and *opponent* nodes (the other player's turn).]
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pyspiel

from config import GAME_NAME, NUM_PLAYERS, SEED


# ── Lightweight containers returned by the env ──────────

class TimeStep(NamedTuple):
    """One step of interaction from a single player's perspective."""
    obs: np.ndarray            # information-state tensor
    legal_actions: List[int]   # indices of legal actions
    legal_actions_mask: np.ndarray  # 1/0 mask over full action space
    reward: float              # reward received *this* step
    done: bool                 # True if the episode is over
    player_id: int             # which player is to act (-1 if terminal)


class EpisodeResult(NamedTuple):
    """Summary returned at the end of an episode."""
    returns: Dict[int, float]  # player_id → cumulative reward
    num_steps: int             # total decision steps in the episode


# ── Main wrapper ────────────────────────────────────────

class PokerEnv:
    """Wraps an OpenSpiel game into a step-based RL environment.

    Usage
    -----
    >>> env = PokerEnv()
    >>> ts = env.reset()
    >>> while not ts.done:
    ...     action = agent.step(ts)          # agent picks an action
    ...     ts = env.step(ts.player_id, action)
    >>> episode_returns = env.get_returns()   # {0: ..., 1: ...}

    [LECTURE: MDPs_1 — The reset() / step() loop mirrors the
     standard MDP interaction protocol used in HW2's GridWorld
     and CartPole environments (Gymnasium API), but extended
     to multi-agent turn-based games.]
    """

    def __init__(
        self,
        game_name: str = GAME_NAME,
        seed: Optional[int] = SEED,
    ):
        self.game: pyspiel.Game = pyspiel.load_game(game_name)
        self.num_players: int = self.game.num_players()
        assert self.num_players == NUM_PLAYERS, (
            f"Expected {NUM_PLAYERS}-player game, got {self.num_players}."
        )
        self.num_actions: int = self.game.num_distinct_actions()
        # Information-state tensor size (input dim for neural nets)
        self.obs_size: int = self.game.information_state_tensor_size()

        self._rng = np.random.RandomState(seed)
        self._state: Optional[pyspiel.State] = None
        self._num_steps: int = 0

    # ── public API ──────────────────────────────────────

    def reset(self) -> TimeStep:
        """Start a new episode; returns the first decision TimeStep.

        [LECTURE: ModelFreeRL — The "reset" initialises the MDP.
         Here it also plays out any initial chance nodes (dealing
         private cards) so the first returned TimeStep is always
         a *decision* node for one of the players.]
        """
        self._state = self.game.new_initial_state()
        self._num_steps = 0
        self._resolve_chance_nodes()
        return self._make_timestep(reward=0.0)

    def step(self, player_id: int, action: int) -> TimeStep:
        """Apply *action* for *player_id*, advance the game, and
        return the next decision TimeStep (or a terminal one).

        Between the action and the returned TimeStep, any
        intervening chance nodes and opponent actions are **not**
        resolved automatically — the caller is expected to call
        step() once per decision node for the *current* player.

        For a simpler "play full episode" loop see `play_episode()`.

        [LECTURE: ModelFreeRL — This is the single-step
         transition  s, a → r, s'  at the heart of all RL
         algorithms (MC, TD, Q-learning).]
        """
        assert not self._state.is_terminal(), "Episode already finished."
        assert self._state.current_player() == player_id, (
            f"Expected player {self._state.current_player()}, got {player_id}."
        )
        assert action in self._state.legal_actions(player_id), (
            f"Action {action} is not legal.  Legal: "
            f"{self._state.legal_actions(player_id)}"
        )

        self._state.apply_action(action)
        self._num_steps += 1

        # Advance past any chance nodes that follow
        self._resolve_chance_nodes()

        # Compute per-step reward
        # (OpenSpiel gives cumulative returns at terminal; we diff later)
        reward = 0.0
        if self._state.is_terminal():
            # final reward for the *acting* player
            reward = self._state.returns()[player_id]
        return self._make_timestep(reward=reward)

    def get_returns(self) -> Dict[int, float]:
        """Return final cumulative payoffs.  Only valid after terminal."""
        assert self._state.is_terminal(), "Episode not finished yet."
        return {pid: self._state.returns()[pid]
                for pid in range(self.num_players)}

    def get_episode_result(self) -> EpisodeResult:
        """Convenience: returns + step count after a finished episode."""
        return EpisodeResult(
            returns=self.get_returns(),
            num_steps=self._num_steps,
        )

    def current_player(self) -> int:
        """Which player is to act next (-1 if chance, -4 if terminal)."""
        return self._state.current_player()

    def is_terminal(self) -> bool:
        return self._state.is_terminal()

    def get_state(self) -> pyspiel.State:
        """Direct access to the underlying OpenSpiel state
        (useful for CFR / exploitability calculations)."""
        return self._state

    def get_game(self) -> pyspiel.Game:
        """Direct access to the underlying OpenSpiel game object."""
        return self.game

    # ── helpers for agents ──────────────────────────────

    def observation_tensor(self, player_id: int) -> np.ndarray:
        """Return the information-state tensor for *player_id*.

        [LECTURE: MDPs_1 / ApproxVI — The "observation" here is
         analogous to the feature vector φ(s) used in HW2's linear
         function approximation.  For neural-net agents (DQN, PPO)
         this tensor *is* the input to the network.]
        """
        return np.array(
            self._state.information_state_tensor(player_id),
            dtype=np.float32,
        )

    def legal_actions(self, player_id: int) -> List[int]:
        return self._state.legal_actions(player_id)

    def legal_actions_mask(self, player_id: int) -> np.ndarray:
        """Binary mask of size `num_actions`; 1 where action is legal.

        [LECTURE: ModelFreeRL — When doing ε-greedy or softmax
         action selection, we must restrict to *legal* actions.
         Masking invalid actions to -∞ before a softmax is the
         standard trick for policy-gradient methods too
         (cf. PolicySearch.pdf, softmax policy parameterisation).]
        """
        mask = np.zeros(self.num_actions, dtype=np.float32)
        for a in self._state.legal_actions(player_id):
            mask[a] = 1.0
        return mask

    # ── internal ────────────────────────────────────────

    def _resolve_chance_nodes(self) -> None:
        """Automatically sample any chance-node actions (card deals).

        [LECTURE: MDPs_1 — Chance nodes are *nature's* moves.
         In an MDP this corresponds to the stochastic transition
         T(s, a, s').  Here the randomness comes from the deck.]
        """
        while (
            not self._state.is_terminal()
            and self._state.is_chance_node()
        ):
            outcomes_with_probs = self._state.chance_outcomes()
            actions, probs = zip(*outcomes_with_probs)
            action = self._rng.choice(actions, p=probs)
            self._state.apply_action(action)

    def _make_timestep(self, reward: float) -> TimeStep:
        if self._state.is_terminal():
            # Return a dummy observation; the episode is over.
            return TimeStep(
                obs=np.zeros(self.obs_size, dtype=np.float32),
                legal_actions=[],
                legal_actions_mask=np.zeros(self.num_actions, dtype=np.float32),
                reward=reward,
                done=True,
                player_id=-1,
            )
        pid = self._state.current_player()
        return TimeStep(
            obs=self.observation_tensor(pid),
            legal_actions=self.legal_actions(pid),
            legal_actions_mask=self.legal_actions_mask(pid),
            reward=reward,
            done=False,
            player_id=pid,
        )


# ── Utility: play a full episode given a dict of agents ─

def play_episode(
    env: PokerEnv,
    agents: Dict[int, "BaseAgent"],  # type: ignore[name-defined]
) -> EpisodeResult:
    """Play one complete episode, letting each agent act in turn.

    Parameters
    ----------
    env : PokerEnv
    agents : dict mapping player_id → agent instance

    Returns
    -------
    EpisodeResult with per-player returns and step count.

    [LECTURE: ModelFreeRL — This is the full "rollout" loop
     used by MC methods (cf. HW2 Section 2, `rollout(env, pi)`).
     The key difference is that we now have *two* agents
     alternating turns in a *partially observable* game.]
    """
    ts = env.reset()
    while not ts.done:
        agent = agents[ts.player_id]
        action = agent.step(ts)
        ts = env.step(ts.player_id, action)
    # Notify agents the episode ended (for experience collection)
    for pid, agent in agents.items():
        agent.on_episode_end(env.get_returns()[pid])
    return env.get_episode_result()
