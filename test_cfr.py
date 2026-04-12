"""
test_cfr.py — Quick test script for the CFR agent implementation.

This script:
1. Creates a CFR agent
2. Trains it for a small number of iterations
3. Evaluates it against a random agent
4. Tests save/load functionality
"""

import pyspiel
from poker_env import PokerEnv
from agents.cfr_agent import CFRAgent
from agents.random_agent import RandomAgent
from evaluate import evaluate_head_to_head
from config import CFRConfig

def main():
    print("="*60)
    print("CFR Agent Test")
    print("="*60)
    
    # Create environment and game
    env = PokerEnv()
    game = pyspiel.load_game("leduc_poker")
    
    print(f"\nGame: {game.get_type().short_name}")
    print(f"Num actions: {env.num_actions}")
    print(f"Observation size: {env.obs_size}")
    
    # Create CFR agent with small number of iterations for testing
    config = CFRConfig(num_iterations=1000)
    cfr_agent = CFRAgent(
        player_id=0,
        num_actions=env.num_actions,
        game=game,
        config=config,
    )
    
    print(f"\nCFR Agent created: {cfr_agent}")
    
    # Train the agent
    print("\n" + "="*60)
    print("Training CFR Agent")
    print("="*60)
    
    metrics = cfr_agent.train(env)
    
    print(f"\nTraining metrics:")
    print(f"  Iterations: {metrics['iterations']}")
    print(f"  Final exploitability: {metrics['final_exploitability']:.6f}")
    print(f"  Exploitability history: {len(metrics['exploitability'])} checkpoints")
    
    # Evaluate against random agent
    print("\n" + "="*60)
    print("Evaluation: CFR vs Random")
    print("="*60)
    
    random_agent = RandomAgent(player_id=1, num_actions=env.num_actions)
    
    result = evaluate_head_to_head(
        cfr_agent,
        random_agent,
        num_episodes=100,
        env=env,
        agent_a_name="CFR",
        agent_b_name="Random",
    )
    
    print("\n" + result.summary())
    
    # Test save/load
    print("\n" + "="*60)
    print("Testing Save/Load")
    print("="*60)
    
    save_path = "models/cfr/test_cfr_policy.pkl"
    
    try:
        cfr_agent.save(save_path)
        print(f"✓ Save successful")
        
        # Create new agent and load
        cfr_agent_loaded = CFRAgent(
            player_id=0,
            num_actions=env.num_actions,
            game=game,
        )
        cfr_agent_loaded.load(save_path)
        print(f"✓ Load successful")
        
        # Quick test that loaded agent works
        result_loaded = evaluate_head_to_head(
            cfr_agent_loaded,
            random_agent,
            num_episodes=10,
            env=env,
            agent_a_name="CFR (loaded)",
            agent_b_name="Random",
        )
        print(f"✓ Loaded agent evaluation: mean return = {result_loaded.mean_return_a:+.4f}")
        
    except Exception as e:
        print(f"✗ Save/Load failed: {e}")
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    
    # Summary
    print("\nSummary:")
    print(f"  CFR trained for {metrics['iterations']} iterations")
    print(f"  Final exploitability: {metrics['final_exploitability']:.6f}")
    print(f"  Win rate vs Random: {result.win_rate_a:.1%}")
    print(f"  Mean return vs Random: {result.mean_return_a:+.4f}")
    
    if result.mean_return_a > 0:
        print("\n✓ CFR agent is beating random agent (expected behavior)")
    else:
        print("\n⚠ CFR agent is not beating random agent (may need more training)")

if __name__ == "__main__":
    main()
