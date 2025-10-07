"""
Basic test to verify GNN model works correctly after optimization
"""
import torch
import sys
sys.path.append('/opendilab/2048GNN/LightZero')

from lzero.model.gnn_stochastic_muzero_model import GNNStochasticMuZeroModel

def test_model_creation_and_forward():
    """Test that model can be created and run forward pass"""
    print("="*60)
    print("Testing GNN Model Creation and Forward Pass")
    print("="*60)
    
    # Test with sparse edge mode (recommended)
    print("\n1. Creating model with edge_mode='sparse'...")
    model = GNNStochasticMuZeroModel(
        observation_shape=(16, 4, 4),
        action_space_size=4,
        chance_space_size=32,
        num_channels=128,
        num_gnn_layers=3,
        grid_size=4,
        edge_mode='sparse',
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
        print(f"✓ Model moved to GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("✓ Model on CPU")
    
    model.eval()
    
    # Test initial_inference
    print("\n2. Testing initial_inference...")
    batch_size = 8
    obs = torch.randn(batch_size, 16, 4, 4).to(device)
    
    try:
        with torch.no_grad():
            output = model.initial_inference(obs)
        print(f"✓ initial_inference successful")
        print(f"  - latent_state shape: {output.latent_state.shape}")
        print(f"  - value shape: {output.value.shape}")
        print(f"  - policy_logits shape: {output.policy_logits.shape}")
    except Exception as e:
        print(f"✗ initial_inference failed: {e}")
        raise
    
    # Test recurrent_inference
    print("\n3. Testing recurrent_inference...")
    latent_state = output.latent_state
    action = torch.randint(0, 4, (batch_size,)).to(device)
    
    try:
        with torch.no_grad():
            output = model.recurrent_inference(latent_state, action)
        print(f"✓ recurrent_inference successful")
        print(f"  - next_latent_state shape: {output.latent_state.shape}")
        print(f"  - reward shape: {output.reward.shape}")
        print(f"  - value shape: {output.value.shape}")
        print(f"  - policy_logits shape: {output.policy_logits.shape}")
    except Exception as e:
        print(f"✗ recurrent_inference failed: {e}")
        raise
    
    # Test project (for SSL)
    print("\n4. Testing project (SSL)...")
    try:
        with torch.no_grad():
            proj = model.project(latent_state, with_grad=False)
        print(f"✓ project successful")
        print(f"  - projection shape: {proj.shape}")
    except Exception as e:
        print(f"✗ project failed: {e}")
        raise
    
    # Count parameters
    print("\n5. Model statistics...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    
    return model

def test_different_edge_modes():
    """Test all three edge modes"""
    print("\n" + "="*60)
    print("Testing Different Edge Modes")
    print("="*60)
    
    for edge_mode in ['adjacent', 'sparse', 'full']:
        print(f"\nTesting edge_mode='{edge_mode}'...")
        
        model = GNNStochasticMuZeroModel(
            observation_shape=(16, 4, 4),
            action_space_size=4,
            chance_space_size=32,
            num_channels=128,
            num_gnn_layers=3,
            grid_size=4,
            edge_mode=edge_mode,
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
            device = 'cuda'
        else:
            device = 'cpu'
        
        model.eval()
        
        # Get edge count
        edge_count = model.representation_network.graph_builder.edge_index.size(1)
        print(f"  - Edge count: {edge_count}")
        
        # Test forward pass
        obs = torch.randn(4, 16, 4, 4).to(device)
        try:
            with torch.no_grad():
                output = model.initial_inference(obs)
            print(f"  ✓ Forward pass successful")
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            raise
    
    print("\n✓ All edge modes work correctly!")

def test_3x3_model():
    """Test 3x3 grid model"""
    print("\n" + "="*60)
    print("Testing 3x3 Grid Model")
    print("="*60)
    
    model = GNNStochasticMuZeroModel(
        observation_shape=(16, 3, 3),
        action_space_size=4,
        chance_space_size=18,  # 9 * 2
        num_channels=96,
        num_gnn_layers=2,
        grid_size=3,
        edge_mode='adjacent',  # Best for 3x3
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    model.eval()
    
    # Get edge count
    edge_count = model.representation_network.graph_builder.edge_index.size(1)
    print(f"\nEdge count for 3x3 grid: {edge_count}")
    
    # Test forward pass
    obs = torch.randn(4, 16, 3, 3).to(device)
    try:
        with torch.no_grad():
            output = model.initial_inference(obs)
        print(f"✓ 3x3 model works correctly")
        print(f"  - latent_state shape: {output.latent_state.shape}")
    except Exception as e:
        print(f"✗ 3x3 model failed: {e}")
        raise

if __name__ == "__main__":
    print("\nGNN Model Basic Tests\n")
    
    # Run all tests
    try:
        model = test_model_creation_and_forward()
        test_different_edge_modes()
        test_3x3_model()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nThe optimized GNN model is ready for training!")
        print("\nKey improvements:")
        print("  ✓ Batched graph processing (no for-loops)")
        print("  ✓ LayerNorm instead of BatchNorm")
        print("  ✓ Flexible edge connectivity modes")
        print("  ✓ reshape instead of view (better compatibility)")
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ TESTS FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
