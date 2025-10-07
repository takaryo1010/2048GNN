import sys
import torch

# Ensure current directory is package root
# This script instantiates the GNN model defined in the 3x3 config and prints info.
from zoo.game_2048.config.stochastic_muzero_2048_gnn_3x3_config import main_config

cfg = main_config
model_cfg = cfg.policy.model

print('Using config model_type:', model_cfg.get('model_type', None))

# Import the GNN model class
from lzero.model.gnn_stochastic_muzero_model import GNNStochasticMuZeroModel

# Instantiate the model using parameters from config (use safe getters)
m = GNNStochasticMuZeroModel(
    observation_shape=tuple(model_cfg.get('observation_shape')),
    action_space_size=int(model_cfg.get('action_space_size')),
    chance_space_size=int(model_cfg.get('chance_space_size')),
    num_channels=int(model_cfg.get('num_channels')),
    num_gnn_layers=int(model_cfg.get('num_gnn_layers')),
    grid_size=int(model_cfg.get('grid_size')),
    value_head_hidden_channels=tuple(model_cfg.get('value_head_hidden_channels')),
    policy_head_hidden_channels=tuple(model_cfg.get('policy_head_hidden_channels')),
    reward_head_hidden_channels=tuple(model_cfg.get('reward_head_hidden_channels')),
    reward_support_size=int(model_cfg.get('reward_support_size')),
    value_support_size=int(model_cfg.get('value_support_size')),
    categorical_distribution=bool(model_cfg.get('categorical_distribution')),
)

print('Model class:', type(m))
print('Representation network class:', type(m.representation_network))
print('GNN module class:', type(m.representation_network.gnn))
print('Total params:', sum(p.numel() for p in m.parameters()))
print('GNN params:', sum(p.numel() for p in m.representation_network.gnn.parameters()))
print('CUDA available:', torch.cuda.is_available())

# print a small summary of gnn submodules
print('GNN submodules:')
for name, mod in m.representation_network.gnn.named_modules():
    print('  ', name, type(mod))
