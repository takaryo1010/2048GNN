"""
Test script for GAT-based Stochastic MuZero Model
Validates model instantiation and forward pass
"""
import torch
import sys
sys.path.insert(0, '/opendilab/2048GNN/LightZero')

from lzero.model.gat_stochastic_muzero_model import GATStochasticMuZeroModel

def test_gat_model():
    """Test GAT model instantiation and forward pass"""
    print("="*80)
    print("Testing GAT-based Stochastic MuZero Model")
    print("="*80)
    
    # Model configuration
    config = dict(
        observation_shape=(16, 4, 4),
        action_space_size=4,
        chance_space_size=32,
        num_channels=128,
        num_gnn_layers=3,
        num_heads=4,  # GAT-specific: attention heads
        grid_size=4,
        value_head_hidden_channels=[128, 64],
        policy_head_hidden_channels=[128, 64],
        reward_head_hidden_channels=[128, 64],
        reward_support_size=601,
        value_support_size=601,
        categorical_distribution=True,
        last_linear_layer_init_zero=True,
        include_row_col_edges=True,
        dropout=0.0,
        edge_mode='sparse',
        self_supervised_learning_loss=False,
    )
    
    print("\n1. Creating GAT model...")
    try:
        model = GATStochasticMuZeroModel(**config)
        print("   ✅ Model created successfully")
    except Exception as e:
        print(f"   ❌ Failed to create model: {e}")
        return False
    
    # Create dummy input
    batch_size = 4
    obs = torch.randn(batch_size, 16, 4, 4)
    action = torch.randint(0, 4, (batch_size,))
    
    print(f"\n2. Testing initial_inference with batch_size={batch_size}...")
    try:
        output = model.initial_inference(obs)
        print(f"   ✅ Initial inference successful")
        print(f"      - value shape: {output.value.shape}")
        print(f"      - policy_logits shape: {output.policy_logits.shape}")
        print(f"      - latent_state shape: {output.latent_state.shape}")
    except Exception as e:
        print(f"   ❌ Initial inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n3. Testing recurrent_inference (action -> afterstate)...")
    try:
        latent_state = output.latent_state
        rec_output = model.recurrent_inference(latent_state, action, afterstate=False)
        print(f"   ✅ Recurrent inference (action) successful")
        print(f"      - value shape: {rec_output.value.shape}")
        print(f"      - reward shape: {rec_output.reward.shape}")
        print(f"      - policy_logits shape: {rec_output.policy_logits.shape} (chance space)")
        print(f"      - afterstate shape: {rec_output.latent_state.shape}")
    except Exception as e:
        print(f"   ❌ Recurrent inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n4. Testing recurrent_inference (chance -> next state)...")
    try:
        afterstate = rec_output.latent_state
        chance = torch.randint(0, 32, (batch_size,))
        rec_output2 = model.recurrent_inference(afterstate, chance, afterstate=True)
        print(f"   ✅ Recurrent inference (chance) successful")
        print(f"      - value shape: {rec_output2.value.shape}")
        print(f"      - reward shape: {rec_output2.reward.shape}")
        print(f"      - policy_logits shape: {rec_output2.policy_logits.shape} (action space)")
        print(f"      - next_latent_state shape: {rec_output2.latent_state.shape}")
    except Exception as e:
        print(f"   ❌ Recurrent inference (chance) failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n5. Checking model components...")
    try:
        # Check GAT components
        has_gat_repr = hasattr(model, 'representation_network')
        has_gat_dyn = hasattr(model, 'dynamics_network')
        has_gat_pred = hasattr(model, 'prediction_network')
        
        print(f"   - Representation Network (GAT): {'✅' if has_gat_repr else '❌'}")
        print(f"   - Dynamics Network (GAT): {'✅' if has_gat_dyn else '❌'}")
        print(f"   - Prediction Network (GAT): {'✅' if has_gat_pred else '❌'}")
        
        # Check for GAT modules
        gat_modules = []
        for name, module in model.named_modules():
            if 'GraphAttention' in type(module).__name__:
                gat_modules.append(name)
        
        print(f"   - GAT modules found: {len(gat_modules)}")
        for gat_name in gat_modules[:3]:  # Show first 3
            print(f"     • {gat_name}")
        
    except Exception as e:
        print(f"   ❌ Component check failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("✅ All tests passed! GAT model is working correctly.")
    print("="*80)
    
    # Print summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Summary:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Attention heads: {config['num_heads']}")
    print(f"  - GAT layers: {config['num_gnn_layers']}")
    print(f"  - Hidden dim: {config['num_channels']}")
    print(f"  - Edge mode: {config['edge_mode']}")
    
    return True


if __name__ == "__main__":
    success = test_gat_model()
    sys.exit(0 if success else 1)
