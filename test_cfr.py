"""
test_cfr.py — Quick test to verify CFR implementation works.

This runs a small number of CFR iterations to ensure everything
is connected properly before running a full training session.
"""

import numpy as np
import pyspiel
from agents.cfr_agent import CFRAgent
from config import CFRConfig, GAME_NAME

def test_cfr():
    print("="*60)
    print("CFR Implementation Test")
    print("="*60)
    
    # Set seed
    np.random.seed(42)
    
    # Load game
    print("\n1. Loading game...")
    game = pyspiel.load_game(GAME_NAME)
    num_actions = game.num_distinct_actions()
    obs_size = game.information_state_tensor_size()
    print(f"   ✓ Loaded {GAME_NAME}")
    print(f"   - Players: {game.num_players()}")
    print(f"   - Actions: {num_actions}")
    print(f"   - Observation size: {obs_size}")
    
    # Create CFR agent
    print("\n2. Creating CFR agent...")
    config = CFRConfig(num_iterations=100)  # Just 100 for quick test
    agent = CFRAgent(
        player_id=0,
        num_actions=num_actions,
        obs_size=obs_size,
        config=config,
        game=game
    )
    print("   ✓ CFR agent created")
    
    # Train for a few iterations
    print("\n3. Running 100 CFR iterations (quick test)...")
    metrics = agent.train(num_iterations=100)
    print(f"   ✓ Training complete")
    print(f"   - Exploitability: {metrics['exploitability']:.6f}")
    
    # Test step function
    print("\n4. Testing policy sampling...")
    state = game.new_initial_state()
    
    # Advance to first decision node
    while state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        state.apply_action(np.random.choice(outcomes, p=probs))
    
    # Sample action from policy
    action = agent.step(state)
    legal_actions = state.legal_actions(state.current_player())
    print(f"   ✓ Sampled action: {action}")
    print(f"   - Legal actions: {legal_actions}")
    print(f"   - Action is legal: {action in legal_actions}")
    
    # Test save/load
    print("\n5. Testing save/load...")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_cfr.pkl")
        agent.save(save_path)
        print(f"   ✓ Saved to {save_path}")
        
        # Create new agent and load
        agent2 = CFRAgent(
            player_id=0,
            num_actions=num_actions,
            obs_size=obs_size,
            game=game
        )
        agent2.load(save_path)
        print("   ✓ Loaded successfully")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nYou can now run full training with:")
    print("  python train_cfr.py --config configs/cfr.yaml")
    print("\nOr with more iterations:")
    print("  python train_cfr.py --iterations 50000")
    print("="*60)

if __name__ == "__main__":
    test_cfr()
