"""
Test script for GNN-based Stochastic MuZero Model
Verifies that forward pass works correctly with dummy inputs
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LightZero'))

import torch
import torch.nn.functional as F

# Import GNN model
from LightZero.lzero.model.gnn_stochastic_muzero_model import GNNStochasticMuZeroModel
from LightZero.lzero.model.gnn_utils import GraphBuilder


def test_graph_builder():
    """Test GraphBuilder functionality"""
    print("=" * 60)
    print("Testing GraphBuilder")
    print("=" * 60)
    
    builder = GraphBuilder(grid_size=4, include_row_col_edges=True)
    
    # Create dummy observation [B, C, H, W]
    batch_size = 2
    obs = torch.randn(batch_size, 16, 4, 4)
    
    # Convert to graph
    node_features, edge_index = builder.obs_to_graph(obs)
    
    print(f"Input obs shape: {obs.shape}")
    print(f"Node features shape: {node_features.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Number of nodes: {node_features.size(1)}")
    print(f"Number of edges: {edge_index.size(1)}")
    print(f"Node feature dim: {node_features.size(2)} (16 channels + 2 positional)")
    
    assert node_features.shape == (batch_size, 16, 18), "Node features shape mismatch"
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, "Edge index shape mismatch"
    
    print("✓ GraphBuilder test passed\n")
    return True


def test_gnn_model_forward():
    """Test GNN model forward pass"""
    print("=" * 60)
    print("Testing GNN Model Forward Pass")
    print("=" * 60)
    
    # Model configuration
    config = dict(
        observation_shape=(16, 4, 4),
        action_space_size=4,
        chance_space_size=32,
        num_channels=128,
        num_gnn_layers=3,
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
    )
    
    # Create model
    model = GNNStochasticMuZeroModel(**config)
    model.eval()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test initial inference
    batch_size = 4
    obs = torch.randn(batch_size, 16, 4, 4)
    
    print(f"\nTesting initial_inference with obs shape: {obs.shape}")
    with torch.no_grad():
        output = model.initial_inference(obs)
    
    print(f"  Latent state shape: {output.latent_state.shape}")
    print(f"  Value shape: {output.value.shape}")
    print(f"  Policy logits shape: {output.policy_logits.shape}")
    print(f"  Reward (list length): {len(output.reward)}")
    
    assert output.latent_state.shape == (batch_size, 128, 4, 4), "Latent state shape mismatch"
    assert output.value.shape == (batch_size, 601), "Value shape mismatch"
    assert output.policy_logits.shape == (batch_size, 4), "Policy shape mismatch"
    
    print("✓ Initial inference test passed")
    
    # Test recurrent inference
    print("\nTesting recurrent_inference")
    latent_state = output.latent_state
    action = F.one_hot(torch.randint(0, 4, (batch_size,)), num_classes=4).float()
    
    print(f"  Input latent state shape: {latent_state.shape}")
    print(f"  Input action shape: {action.shape}")
    
    with torch.no_grad():
        output = model.recurrent_inference(latent_state, action, afterstate=False)
    
    print(f"  Next latent state shape: {output.latent_state.shape}")
    print(f"  Reward shape: {output.reward.shape}")
    print(f"  Value shape: {output.value.shape}")
    print(f"  Policy logits shape: {output.policy_logits.shape}")
    
    assert output.latent_state.shape == (batch_size, 128, 4, 4), "Next latent state shape mismatch"
    assert output.reward.shape == (batch_size, 601), "Reward shape mismatch"
    assert output.value.shape == (batch_size, 601), "Value shape mismatch"
    assert output.policy_logits.shape == (batch_size, 4), "Policy shape mismatch"
    
    print("✓ Recurrent inference test passed")
    
    # Test afterstate dynamics
    print("\nTesting afterstate dynamics")
    chance_action = F.one_hot(torch.randint(0, 32, (batch_size,)), num_classes=32).float()
    
    with torch.no_grad():
        output = model.recurrent_inference(latent_state, chance_action, afterstate=True)
    
    print(f"  Afterstate latent shape: {output.latent_state.shape}")
    print(f"  Afterstate policy logits shape: {output.policy_logits.shape}")
    
    assert output.latent_state.shape == (batch_size, 128, 4, 4), "Afterstate shape mismatch"
    assert output.policy_logits.shape == (batch_size, 32), "Afterstate policy shape mismatch"
    
    print("✓ Afterstate dynamics test passed\n")
    
    return True


def test_gradient_flow():
    """Test that gradients flow through the model"""
    print("=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)
    
    model = GNNStochasticMuZeroModel(
        observation_shape=(16, 4, 4),
        action_space_size=4,
        chance_space_size=32,
        num_channels=64,  # Smaller for faster test
        num_gnn_layers=2,
        grid_size=4,
    )
    model.train()
    
    obs = torch.randn(2, 16, 4, 4, requires_grad=True)
    
    # Forward pass
    output = model.initial_inference(obs)
    
    # Compute dummy loss
    loss = output.value.sum() + output.policy_logits.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "No gradients found in model parameters"
    print("✓ Gradients flow correctly through the model\n")
    
    return True


def test_cuda_compatibility():
    """Test CUDA compatibility if available"""
    print("=" * 60)
    print("Testing CUDA Compatibility")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA test")
        return True
    
    print("CUDA is available, testing GPU inference")
    
    model = GNNStochasticMuZeroModel(
        observation_shape=(16, 4, 4),
        action_space_size=4,
        chance_space_size=32,
        num_channels=64,
        num_gnn_layers=2,
    )
    model = model.cuda()
    model.eval()
    
    obs = torch.randn(2, 16, 4, 4).cuda()
    
    with torch.no_grad():
        output = model.initial_inference(obs)
    
    assert output.latent_state.is_cuda, "Output should be on CUDA"
    print("✓ CUDA inference works correctly\n")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("GNN Stochastic MuZero Model Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        ("GraphBuilder", test_graph_builder),
        ("Model Forward Pass", test_gnn_model_forward),
        ("Gradient Flow", test_gradient_flow),
        ("CUDA Compatibility", test_cuda_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with error:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
