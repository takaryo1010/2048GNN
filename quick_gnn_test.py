"""
Quick test to verify GNN model training loop works
"""
import sys
sys.path.append('/opendilab/2048GNN/LightZero')

import torch
from lzero.entry import train_muzero
from zoo.game_2048.config.stochastic_muzero_2048_gnn_config import main_config, create_config

# Modify config for quick test
main_config.policy.num_simulations = 10  # Reduce MCTS simulations
main_config.policy.batch_size = 32  # Smaller batch
main_config.policy.num_unroll_steps = 2  # Fewer unroll steps
create_config.env_manager.n_evaluator_episode = 2
create_config.collector_env_num = 2
create_config.evaluator_env_num = 2

if __name__ == "__main__":
    print("=" * 80)
    print("Quick GNN Stochastic MuZero Test")
    print("=" * 80)
    print(f"Model type: {main_config.policy.model.model_type}")
    print(f"MCTS simulations: {main_config.policy.num_simulations}")
    print(f"Batch size: {main_config.policy.batch_size}")
    print(f"Unroll steps: {main_config.policy.num_unroll_steps}")
    print("=" * 80)
    
    train_muzero([main_config, create_config], seed=0, max_env_step=100)
    print("\n" + "=" * 80)
    print("Quick test completed successfully!")
    print("=" * 80)
