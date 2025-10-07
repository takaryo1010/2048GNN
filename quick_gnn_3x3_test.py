"""
Quick test for 3x3 GNN Stochastic MuZero
"""
import sys
sys.path.append('/opendilab/2048GNN/LightZero')

import torch
from lzero.entry import train_muzero
from zoo.game_2048.config.stochastic_muzero_2048_gnn_3x3_config import main_config, create_config

# Modify config for quick test
main_config.policy.num_simulations = 10  # Reduce MCTS simulations
main_config.policy.batch_size = 32  # Smaller batch
main_config.policy.num_unroll_steps = 2  # Fewer unroll steps
create_config.env_manager.n_evaluator_episode = 2
main_config.policy.collector_env_num = 2
main_config.policy.evaluator_env_num = 2
main_config.env.collector_env_num = 2
main_config.env.evaluator_env_num = 2
main_config.env.n_evaluator_episode = 2

if __name__ == "__main__":
    print("=" * 80)
    print("Quick GNN Stochastic MuZero Test - 3x3 Grid")
    print("=" * 80)
    print(f"Model type: {main_config.policy.model.model_type}")
    print(f"Grid size: {main_config.policy.model.grid_size}")
    print(f"Observation shape: {main_config.policy.model.observation_shape}")
    print(f"Chance space size: {main_config.policy.model.chance_space_size}")
    print(f"MCTS simulations: {main_config.policy.num_simulations}")
    print(f"Batch size: {main_config.policy.batch_size}")
    print("=" * 80)
    
    try:
        train_muzero([main_config, create_config], seed=0, max_env_step=100)
        print("\n" + "=" * 80)
        print("3x3 test completed successfully!")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Error occurred: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
