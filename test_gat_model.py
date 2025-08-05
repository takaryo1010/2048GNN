#!/usr/bin/env python3
"""
Test script for GAT-based MuZero model on 2048 game
"""

import sys
import os
import torch
import numpy as np

# Add LightZero to Python path
sys.path.append('/opendilab/LightZero')

def test_gat_model():
    """Test if GAT model can be imported and instantiated"""
    print("Testing GAT model import and instantiation...")
    
    try:
        from lzero.model.gat_muzero_model import GATMuZeroModel
        print("✓ Successfully imported GATMuZeroModel")
        
        # Test model instantiation
        model = GATMuZeroModel(
            observation_shape=(16, 4, 4),
            action_space_size=4,
            num_heads=4,
            hidden_channels=64,
            num_gat_layers=3,
            state_dim=256,
        )
        print("✓ Successfully instantiated GATMuZeroModel")
        
        # Test forward pass
        batch_size = 2
        obs = torch.randn(batch_size, 16, 4, 4)
        
        # Test initial inference
        output = model.initial_inference(obs)
        print(f"✓ Initial inference successful")
        print(f"  - Value shape: {output.value.shape}")
        print(f"  - Policy logits shape: {output.policy_logits.shape}")
        print(f"  - Latent state shape: {output.latent_state.shape}")
        
        # Test recurrent inference
        action = torch.tensor([0, 1])
        output = model.recurrent_inference(output.latent_state, action)
        print(f"✓ Recurrent inference successful")
        print(f"  - Reward shape: {output.reward.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test if config can be loaded"""
    print("\nTesting configuration loading...")
    
    try:
        # Change to the config directory
        config_dir = '/opendilab/LightZero/zoo/game_2048/config'
        os.chdir(config_dir)
        
        sys.path.append(config_dir)
        
        from gat_2048_config import main_config, create_config
        print("✓ Successfully loaded GAT config")
        print(f"  - Model type: {main_config.policy.model.model_type}")
        print(f"  - Grid size: {main_config.policy.model.observation_shape}")
        print(f"  - Num heads: {main_config.policy.model.num_heads}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_env():
    """Test if environment can be created"""
    print("\nTesting environment creation...")
    
    try:
        from zoo.game_2048.envs.game_2048_env import Game2048Env
        from easydict import EasyDict
        
        env_config = EasyDict({
            'env_id': 'game_2048',
            'render_mode': None,
            'replay_format': 'gif',
            'replay_name_suffix': 'test',
            'replay_path': None,
            'act_scale': True,
            'channel_last': False,
            'obs_type': 'dict_encoded_board',
            'reward_normalize': False,
            'reward_norm_scale': 100,
            'reward_type': 'raw',
            'max_tile': 65536,
            'delay_reward_step': 0,
            'prob_random_agent': 0.0,
            'max_episode_steps': 1000000,
            'is_collect': True,
            'ignore_legal_actions': True,
            'need_flatten': False,
            'num_of_possible_chance_tile': 2,
            'possible_tiles': np.array([2, 4]),
            'tile_probabilities': np.array([0.9, 0.1]),
        })
        
        env = Game2048Env(env_config)
        print("✓ Successfully created environment")
        
        # Test reset
        obs = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  - Observation keys: {obs.keys() if isinstance(obs, dict) else 'Not a dict'}")
        
        if isinstance(obs, dict) and 'observation' in obs:
            obs_shape = obs['observation'].shape
            print(f"  - Observation shape: {obs_shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("GAT-based MuZero 2048 Test Suite")
    print("=" * 50)
    
    success = True
    success &= test_gat_model()
    success &= test_config()
    success &= test_env()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! GAT model is ready for training.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    print("\nTo train the model, run:")
    print("cd /opendilab/LightZero/zoo/game_2048/config")
    print("python gat_2048_config.py")
