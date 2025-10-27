"""
最適化版GNNモデルの動作確認スクリプト
"""
import sys
sys.path.insert(0, '/opendilab/2048GNN/LightZero')

import torch
from easydict import EasyDict
from lzero.model.gnn_stochastic_muzero_model_optimized import GNNStochasticMuZeroModelOptimized
from lzero.policy.stochastic_muzero import StochasticMuZeroPolicy

def test_model_creation():
    """モデル作成テスト"""
    print("="*60)
    print("Test 1: Model Creation")
    print("="*60)
    
    model = GNNStochasticMuZeroModelOptimized(
        observation_shape=(16, 4, 4),
        action_space_size=4,
        chance_space_size=32,
        num_channels=128,
        num_gnn_layers=3,
        grid_size=4,
    )
    
    print("✅ Model created successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model

def test_forward_pass(model):
    """Forward passテスト"""
    print("\n" + "="*60)
    print("Test 2: Forward Pass")
    print("="*60)
    
    batch_size = 4
    obs = torch.randn(batch_size, 16, 4, 4)
    
    # Initial inference
    output = model.initial_inference(obs)
    print(f"✅ initial_inference successful")
    print(f"   latent_state shape: {output.latent_state.shape}")
    print(f"   Expected: [{batch_size}, 16, 128] (B, N, C)")
    print(f"   value shape: {output.value.shape}")
    print(f"   policy_logits shape: {output.policy_logits.shape}")
    
    # Recurrent inference (action)
    action = torch.randint(0, 4, (batch_size,))
    output2 = model.recurrent_inference(output.latent_state, action, afterstate=False)
    print(f"\n✅ recurrent_inference (action) successful")
    print(f"   latent_state shape: {output2.latent_state.shape}")
    
    # Recurrent inference (chance)
    chance = torch.randint(0, 32, (batch_size,))
    output3 = model.recurrent_inference(output2.latent_state, chance, afterstate=True)
    print(f"\n✅ recurrent_inference (chance) successful")
    print(f"   latent_state shape: {output3.latent_state.shape}")
    
    return True

def test_policy_integration():
    """Policyとの統合テスト"""
    print("\n" + "="*60)
    print("Test 3: Policy Integration")
    print("="*60)
    
    cfg = EasyDict({
        'type': 'stochastic_muzero',
        'model': {
            'model_type': 'gnn_optimized',
            'observation_shape': (16, 4, 4),
            'action_space_size': 4,
            'chance_space_size': 32,
            'num_channels': 128,
            'num_gnn_layers': 3,
            'grid_size': 4,
            'value_head_hidden_channels': [128, 64],
            'policy_head_hidden_channels': [128, 64],
            'reward_head_hidden_channels': [128, 64],
            'reward_support_size': 601,
            'value_support_size': 601,
            'categorical_distribution': True,
            'self_supervised_learning_loss': False,
            'image_channel': 16,
            'frame_stack_num': 1,
        },
        'cuda': False,
        'learn': {
            'update_per_collect': 200,
            'batch_size': 256,
        },
        'collect': {
            'n_episode': 8,
            'collector_env_num': 8,
        },
        'eval': {
            'evaluator_env_num': 3,
        }
    })
    
    try:
        policy = StochasticMuZeroPolicy(cfg, enable_field=['learn', 'collect', 'eval'])
        print("✅ Policy created successfully")
        print(f"   Model type: {policy._cfg.model.model_type}")
        return True
    except Exception as e:
        print(f"❌ Policy creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mcts_utils():
    """MCTS utilsの互換性テスト"""
    print("\n" + "="*60)
    print("Test 4: MCTS Utils Compatibility")
    print("="*60)
    
    from lzero.mcts.utils import prepare_observation
    import numpy as np
    
    # テスト観測データ
    obs_list = [np.random.randn(16, 4, 4).astype(np.float32) for _ in range(4)]
    
    try:
        result = prepare_observation(obs_list, model_type='gnn_optimized')
        print(f"✅ prepare_observation successful")
        print(f"   Result shape: {result.shape}")
        return True
    except Exception as e:
        print(f"❌ prepare_observation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("GNN Optimized Model - Comprehensive Test")
    print("="*60)
    
    results = []
    
    try:
        # Test 1: Model creation
        model = test_model_creation()
        results.append(("Model Creation", True))
        
        # Test 2: Forward pass
        forward_ok = test_forward_pass(model)
        results.append(("Forward Pass", forward_ok))
        
        # Test 3: Policy integration
        policy_ok = test_policy_integration()
        results.append(("Policy Integration", policy_ok))
        
        # Test 4: MCTS utils
        mcts_ok = test_mcts_utils()
        results.append(("MCTS Utils", mcts_ok))
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("General", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("🎉 All tests passed! Ready for training.")
    else:
        print("⚠️  Some tests failed. Please fix the issues.")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
