"""
Debug script to check action format in dynamics network
"""
import sys
sys.path.insert(0, '/opendilab/2048GNN/LightZero')

import torch
import torch.nn.functional as F
from lzero.model.gnn_stochastic_muzero_model import GNNStochasticMuZeroModel

# Create model
model = GNNStochasticMuZeroModel(
    observation_shape=(16, 4, 4),
    action_space_size=4,
    chance_space_size=32,
    num_channels=64,
    num_gnn_layers=2,
)
model.eval()

# Test inputs
obs = torch.randn(2, 16, 4, 4)
print("Testing GNN model...")

# Initial inference
with torch.no_grad():
    output = model.initial_inference(obs)
    print(f"✓ Initial inference OK")
    print(f"  Latent state: {output.latent_state.shape}")
    
    latent_state = output.latent_state
    
    # Test different action formats
    print("\nTesting action formats:")
    
    # Format 1: Index (what we expect from MCTS)
    action_idx = torch.tensor([0, 1])  # [B]
    print(f"  Action index format: {action_idx.shape} = {action_idx}")
    try:
        output = model.recurrent_inference(latent_state, action_idx, afterstate=False)
        print(f"  ✓ Index format works!")
    except Exception as e:
        print(f"  ✗ Index format failed: {e}")
    
    # Format 2: Index with extra dim
    action_idx2 = torch.tensor([[0], [1]])  # [B, 1]
    print(f"  Action index+dim format: {action_idx2.shape} = {action_idx2.flatten()}")
    try:
        output = model.recurrent_inference(latent_state, action_idx2, afterstate=False)
        print(f"  ✓ Index+dim format works!")
    except Exception as e:
        print(f"  ✗ Index+dim format failed: {e}")
    
    # Format 3: One-hot
    action_onehot = F.one_hot(torch.tensor([0, 1]), num_classes=4).float()  # [B, A]
    print(f"  Action one-hot format: {action_onehot.shape}")
    try:
        output = model.recurrent_inference(latent_state, action_onehot, afterstate=False)
        print(f"  ✓ One-hot format works!")
    except Exception as e:
        print(f"  ✗ One-hot format failed: {e}")
    
    # Format 4: What MCTS might send (chance/afterstate)
    chance_idx = torch.tensor([5, 10])  # [B] - indices for chance actions
    print(f"\n  Testing afterstate with chance index: {chance_idx.shape} = {chance_idx}")
    try:
        output = model.recurrent_inference(latent_state, chance_idx, afterstate=True)
        print(f"  ✓ Afterstate with index works!")
    except Exception as e:
        print(f"  ✗ Afterstate with index failed: {e}")

print("\n✅ All tests completed")
