"""
evaluate_cfr.py — Evaluation utilities for CFR agents.

CFR agents work directly with OpenSpiel states rather than
TimeStep wrappers, so they need special evaluation logic.

Usage:
    python evaluate_cfr.py --cfr_model models/cfr/leduc_cfr.pkl
    python evaluate_cfr.py --vs random --episodes 1000
"""

import argparse
import numpy as np
import pyspiel
from typing import Dict

from agents.cfr_agent import CFRAgent
from agents.random_agent import RandomAgent
from config import GAME_NAME, SEED


def play_episode_cfr(game: pyspiel.Game, agents: Dict[int, object]) -> Dict[int, float]:
    """
    Play one episode with CFR-compatible agents.
    
    Unlike poker_env which uses TimeStep wrappers, this works
    directly with OpenSpiel states for CFR compatibility.
    
    Parameters
    ----------
    game : pyspiel.Game
        The game instance
    agents : dict
        Mapping from player_id to agent
        
    Returns
    -------
    dict — final returns for each player
    """
    state = game.new_initial_state()
    
    while not state.is_terminal():
        if state.is_chance_node():
            # Handle chance nodes (card dealing)
            outcomes, probs = zip(*state.chance_outcomes())
            action = np.random.choice(outcomes, p=probs)
            state.apply_action(action)
        else:
            # Player decision node
            cur_player = state.current_player()
            agent = agents[cur_player]
            
            # Different agents have different interfaces
            if isinstance(agent, CFRAgent):
                action = agent.step(state)
            elif hasattr(agent, 'step_with_state'):
                action = agent.step_with_state(state)
            else:
                # Random agent - just pick a legal action
                action = np.random.choice(state.legal_actions(cur_player))
            
            state.apply_action(action)
    
    # Return final payoffs
    return {i: state.returns()[i] for i in range(game.num_players())}


def evaluate_cfr_vs_random(
    cfr_agent: CFRAgent,
    num_episodes: int = 1000,
    game: pyspiel.Game = None,
    verbose: bool = True
) -> dict:
    """
    Evaluate CFR agent against a random opponent.
    
    Parameters
    ----------
    cfr_agent : CFRAgent
        The trained CFR agent
    num_episodes : int
        Number of games to play
    game : pyspiel.Game, optional
        Game instance (creates new one if None)
    verbose : bool
        Whether to print results
        
    Returns
    -------
    dict — evaluation statistics
    """
    if game is None:
        game = pyspiel.load_game(GAME_NAME)
    
    # CFR agent plays as player 0, random as player 1
    random_agent = RandomAgent(player_id=1, num_actions=game.num_distinct_actions())
    agents = {0: cfr_agent, 1: random_agent}
    
    returns_cfr = []
    returns_random = []
    
    if verbose:
        print(f"Evaluating CFR vs Random over {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        result = play_episode_cfr(game, agents)
        returns_cfr.append(result[0])
        returns_random.append(result[1])
        
        if verbose and (ep + 1) % (num_episodes // 10) == 0:
            print(f"  Progress: {ep + 1}/{num_episodes} episodes")
    
    returns_cfr = np.array(returns_cfr)
    returns_random = np.array(returns_random)
    
    stats = {
        'num_episodes': num_episodes,
        'cfr_mean_return': float(returns_cfr.mean()),
        'cfr_std_return': float(returns_cfr.std()),
        'cfr_win_rate': float((returns_cfr > 0).mean()),
        'random_mean_return': float(returns_random.mean()),
        'random_win_rate': float((returns_random > 0).mean()),
        'draw_rate': float((returns_cfr == 0).mean()),
    }
    
    if verbose:
        print("\n" + "="*60)
        print("CFR vs Random Evaluation Results")
        print("="*60)
        print(f"Episodes: {stats['num_episodes']}")
        print(f"\nCFR Agent:")
        print(f"  Mean return: {stats['cfr_mean_return']:+.4f}")
        print(f"  Std return:  {stats['cfr_std_return']:.4f}")
        print(f"  Win rate:    {stats['cfr_win_rate']:.2%}")
        print(f"\nRandom Agent:")
        print(f"  Mean return: {stats['random_mean_return']:+.4f}")
        print(f"  Win rate:    {stats['random_win_rate']:.2%}")
        print(f"\nDraw rate:     {stats['draw_rate']:.2%}")
        print("="*60)
    
    return stats


def evaluate_exploitability(cfr_agent: CFRAgent, verbose: bool = True) -> float:
    """
    Compute exploitability of the CFR policy.
    
    Exploitability = 0 means perfect Nash equilibrium.
    Lower values are better.
    
    [LECTURE: DeepRL — Exploitability measures how much better
     an optimal adversary can do against this policy compared to
     Nash equilibrium play.  It's the gold standard for evaluating
     poker agents in small games.]
    """
    exploit = cfr_agent.get_exploitability()
    
    if verbose:
        print("\n" + "="*60)
        print("Exploitability Analysis")
        print("="*60)
        print(f"Exploitability: {exploit:.6f}")
        print(f"\nInterpretation:")
        if exploit < 0.001:
            print("  ✓ Excellent - very close to Nash equilibrium")
        elif exploit < 0.01:
            print("  ✓ Good - reasonably close to Nash equilibrium")
        elif exploit < 0.1:
            print("  ○ Fair - some exploitable weaknesses")
        else:
            print("  ✗ Poor - highly exploitable")
        print("\n(Lower is better; 0 = perfect Nash equilibrium)")
        print("="*60)
    
    return exploit


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CFR agent"
    )
    parser.add_argument(
        "--cfr_model",
        type=str,
        default="models/cfr/leduc_cfr.pkl",
        help="Path to trained CFR model"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--vs",
        type=str,
        default="random",
        choices=["random"],
        help="Opponent type"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed"
    )
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    
    # Load game
    game = pyspiel.load_game(GAME_NAME)
    num_actions = game.num_distinct_actions()
    obs_size = game.information_state_tensor_size()
    
    print("\n" + "="*60)
    print("CFR Agent Evaluation")
    print("="*60)
    print(f"Game: {GAME_NAME}")
    print(f"Model: {args.cfr_model}")
    print(f"Opponent: {args.vs}")
    print("="*60 + "\n")
    
    # Load CFR agent
    cfr_agent = CFRAgent(
        player_id=0,
        num_actions=num_actions,
        obs_size=obs_size,
        game=game
    )
    
    try:
        cfr_agent.load(args.cfr_model)
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.cfr_model}")
        print("Please train a CFR model first using: python train_cfr.py")
        return
    
    # Evaluate exploitability (game-theoretic quality)
    evaluate_exploitability(cfr_agent)
    
    # Evaluate against opponent
    if args.vs == "random":
        print()  # blank line
        evaluate_cfr_vs_random(cfr_agent, num_episodes=args.episodes, game=game)


if __name__ == "__main__":
    main()
