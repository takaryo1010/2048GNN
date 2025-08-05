from typing import Optional, Tuple, Union, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import MZNetworkOutput
from .utils import renormalize, get_params_mean, get_dynamic_mean, get_reward_mean


class GridToGraphConverter:
    """Convert 2048 grid to graph representation for StochasticMuZero"""
    
    def __init__(self, grid_size: int = 4):
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size
        self.edge_index = self._create_edge_index()
    
    def _create_edge_index(self):
        """Create edge connections for grid graph (4-connectivity)"""
        edges = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                node_id = i * self.grid_size + j
                
                # Right neighbor
                if j + 1 < self.grid_size:
                    neighbor_id = i * self.grid_size + (j + 1)
                    edges.append([node_id, neighbor_id])
                    edges.append([neighbor_id, node_id])  # Bidirectional
                
                # Bottom neighbor
                if i + 1 < self.grid_size:
                    neighbor_id = (i + 1) * self.grid_size + j
                    edges.append([node_id, neighbor_id])
                    edges.append([neighbor_id, node_id])  # Bidirectional
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def grid_to_graph(self, grid_batch):
        """
        Convert batch of grids to batch of graphs
        Args:
            grid_batch: (batch_size, channels, height, width) - encoded board representation
        Returns:
            PyTorch Geometric Batch object
        """
        batch_size = grid_batch.shape[0]
        channels = grid_batch.shape[1]
        device = grid_batch.device
        
        # Flatten spatial dimensions to create node features
        node_features = grid_batch.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
        
        graphs = []
        for i in range(batch_size):
            graph = Data(
                x=node_features[i],
                edge_index=self.edge_index.clone().to(device),
            )
            graphs.append(graph)
        
        return Batch.from_data_list(graphs)


class GATRepresentationNetwork(nn.Module):
    """Graph Attention Network for representation learning in StochasticMuZero"""
    
    def __init__(
        self,
        grid_size: int = 4,
        input_channels: int = 16,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.converter = GridToGraphConverter(grid_size)
        
        # Input projection
        self.input_proj = nn.Linear(input_channels, hidden_channels)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_channels = hidden_channels
            else:
                in_channels = hidden_channels * num_heads
            
            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    concat=(i < num_layers - 1)
                )
            )
        
        # Output projection
        final_dim = hidden_channels
        self.output_proj = MLP(
            in_channels=final_dim,
            hidden_channels=output_dim // 2,
            out_channels=output_dim,
            layer_num=2,
            activation=nn.ReLU(),
            norm_type='LN'
        )
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: tensor (batch_size, channels, height, width) - encoded board representation
        Returns:
            (batch_size, output_dim) - graph-level representation
        """
        # Ensure correct dtype
        if x.dtype != torch.float32:
            x = x.float()
        
        return self._process_tensor_input(x)
    
    def _process_tensor_input(self, x):
        """Process tensor input by converting to graph and applying GAT"""
        # Convert grid to graph
        graph_batch = self.converter.grid_to_graph(x)
    
        # Input projection
        h = self.input_proj(graph_batch.x)
        h = self.activation(h)
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            h = gat_layer(h, graph_batch.edge_index)
            if i < len(self.gat_layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        
        # Global pooling to get graph-level representation
        graph_repr = global_mean_pool(h, graph_batch.batch)
        
        # Final projection
        output = self.output_proj(graph_repr)
        
        return output


class GATChanceEncoder(nn.Module):
    """GAT-based chance encoder for StochasticMuZero"""
    
    def __init__(
        self,
        grid_size: int = 4,
        chance_space_size: int = 32,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.chance_space_size = chance_space_size
        
        # Chance embedding
        self.chance_embedding = nn.Embedding(chance_space_size, hidden_channels)
        
        # GAT for chance processing
        self.gat_chance = GATRepresentationNetwork(
            grid_size=grid_size,
            input_channels=16 + hidden_channels // 4,  # Original + chance features
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # Chance fusion
        self.chance_fusion = MLP(
            in_channels=output_dim + hidden_channels,
            hidden_channels=output_dim,
            out_channels=output_dim,
            layer_num=2,
            activation=nn.ReLU(),
            norm_type='LN'
        )
    
    def forward(self, state, chance):
        """
        Args:
            state: (batch_size, channels, height, width) - current state
            chance: (batch_size,) - chance variable
        Returns:
            (batch_size, output_dim) - chance-conditioned state representation
        """
        # Encode chance
        chance_embed = self.chance_embedding(chance)
        
        # Expand chance embedding to spatial dimensions
        batch_size = state.shape[0]
        chance_spatial = chance_embed.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, -1, self.grid_size, self.grid_size
        )
        
        # Concatenate with state (reduce channels to fit)
        chance_reduced = F.adaptive_avg_pool2d(chance_spatial, (self.grid_size, self.grid_size))
        chance_reduced = chance_reduced[:, :state.shape[1]//4]  # Reduce to 1/4 of original channels
        
        state_chance = torch.cat([state, chance_reduced], dim=1)
        
        # Process through GAT
        state_repr = self.gat_chance(state_chance)
        
        # Fuse with global chance representation
        fused = torch.cat([state_repr, chance_embed], dim=-1)
        output = self.chance_fusion(fused)
        
        return output


class GATAfterstateDynamicsNetwork(nn.Module):
    """GAT-based afterstate dynamics network for StochasticMuZero"""
    
    def __init__(
        self,
        grid_size: int = 4,
        state_dim: int = 256,
        action_space_size: int = 4,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.state_dim = state_dim
        self.grid_size = grid_size
        
        # Action encoding
        self.action_embedding = nn.Embedding(action_space_size, state_dim // 4)
        
        # State-action fusion
        fusion_dim = state_dim + state_dim // 4
        self.state_action_fusion = MLP(
            in_channels=fusion_dim,
            hidden_channels=hidden_channels * 2,
            out_channels=state_dim,
            layer_num=3,
            activation=nn.ReLU(),
            norm_type='LN'
        )
        
        # Convert to grid for GAT processing
        self.to_grid_proj = nn.Linear(state_dim, grid_size * grid_size * 16)
        
        # GAT for afterstate dynamics
        self.gat_afterstate = GATRepresentationNetwork(
            grid_size=grid_size,
            input_channels=16,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=state_dim,
            dropout=dropout
        )
    
    def forward(self, state, action):
        """
        Args:
            state: (batch_size, state_dim)
            action: (batch_size,) - discrete actions
        Returns:
            afterstate: (batch_size, state_dim) - deterministic afterstate
        """
        # Encode action
        action_embed = self.action_embedding(action)
        
        # Ensure proper dimensions
        if state.dim() > 2:
            state = state.view(state.size(0), -1)
        if action_embed.dim() > 2:
            action_embed = action_embed.view(action_embed.size(0), -1)
        
        # Fuse state and action
        state_action = torch.cat([state, action_embed], dim=-1)
        fused = self.state_action_fusion(state_action)
        
        # Convert to grid representation
        grid_repr = self.to_grid_proj(fused)
        grid_repr = grid_repr.view(-1, 16, self.grid_size, self.grid_size)
        
        # Apply GAT dynamics
        afterstate = self.gat_afterstate(grid_repr)
        
        return afterstate


@MODEL_REGISTRY.register('GATStochasticMuZeroModel')
class GATStochasticMuZeroModel(nn.Module):
    """
    StochasticMuZero model using Graph Attention Networks for 2048 game
    Supports both 3x3 and 4x4 board sizes with stochastic environment modeling
    """
    
    def __init__(
        self,
        observation_shape: SequenceType = (16, 4, 4),
        action_space_size: int = 4,
        chance_space_size: int = 32,
        num_heads: int = 4,
        hidden_channels: int = 64,
        num_gat_layers: int = 3,
        state_dim: int = 256,
        value_head_channels: int = 16,
        policy_head_channels: int = 16,
        value_head_hidden_channels: SequenceType = [32],
        policy_head_hidden_channels: SequenceType = [32],
        reward_support_size: int = 601,
        value_support_size: int = 601,
        categorical_distribution: bool = True,
        last_linear_layer_init_zero: bool = True,
        state_norm: bool = False,
        dropout: float = 0.1,
        *args,
        **kwargs
    ):
        super().__init__()
        
        # Determine grid size from observation shape
        self.grid_size = observation_shape[-1]
        assert self.grid_size in [3, 4], f"Only 3x3 and 4x4 grids supported, got {self.grid_size}x{self.grid_size}"
        
        self.action_space_size = action_space_size
        self.chance_space_size = chance_space_size
        self.state_dim = state_dim
        self.categorical_distribution = categorical_distribution
        self.state_norm = state_norm
        
        if categorical_distribution:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size
        else:
            self.reward_support_size = 1
            self.value_support_size = 1
        
        # Representation network (encoder)
        self.representation_network = GATRepresentationNetwork(
            grid_size=self.grid_size,
            input_channels=observation_shape[0],
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_layers=num_gat_layers,
            output_dim=state_dim,
            dropout=dropout
        )
        
        # Afterstate dynamics network (deterministic)
        self.afterstate_dynamics_network = GATAfterstateDynamicsNetwork(
            grid_size=self.grid_size,
            state_dim=state_dim,
            action_space_size=action_space_size,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_layers=num_gat_layers - 1,
            dropout=dropout
        )
        
        # Chance encoder (stochastic)
        self.chance_encoder = GATChanceEncoder(
            grid_size=self.grid_size,
            chance_space_size=chance_space_size,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_layers=num_gat_layers - 1,
            output_dim=state_dim,
            dropout=dropout
        )
        
        # Reward prediction from afterstate
        self.reward_head_afterstate = MLP(
            in_channels=state_dim,
            hidden_channels=value_head_hidden_channels[0],
            out_channels=self.reward_support_size,
            layer_num=len(value_head_hidden_channels) + 1,
            activation=nn.ReLU(),
            norm_type='LN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        
        # Value and policy heads
        self.value_head = MLP(
            in_channels=state_dim,
            hidden_channels=value_head_hidden_channels[0],
            out_channels=self.value_support_size,
            layer_num=len(value_head_hidden_channels) + 1,
            activation=nn.ReLU(),
            norm_type='LN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        
        self.policy_head = MLP(
            in_channels=state_dim,
            hidden_channels=policy_head_hidden_channels[0],
            out_channels=action_space_size,
            layer_num=len(policy_head_hidden_channels) + 1,
            activation=nn.ReLU(),
            norm_type='LN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
    
    def initial_inference(self, observation):
        """Initial inference for MCTS"""
        batch_size = observation.shape[0]
        device = observation.device
        
        # Encode observation to latent state
        latent_state = self._representation(observation)
        
        # Predict value and policy
        value, policy_logits = self._prediction(latent_state)
        
        # Return zero reward for initial inference
        if self.categorical_distribution:
            if not self.training:
                reward = [0. for _ in range(batch_size)]
            else:
                reward = torch.zeros(batch_size, self.reward_support_size).to(device)
        else:
            reward = [0. for _ in range(batch_size)]
        
        return MZNetworkOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            latent_state=latent_state
        )
    
    def recurrent_inference(self, latent_state, action, chance):
        """Recurrent inference for MCTS expansion with chance"""
        # Apply afterstate dynamics (deterministic)
        afterstate = self._afterstate_dynamics(latent_state, action)
        
        # Predict reward from afterstate
        reward = self._afterstate_reward(afterstate)
        
        # Apply chance encoding (stochastic)
        next_latent_state = self._chance_encoding(afterstate, chance)
        
        # Predict value and policy for next state
        value, policy_logits = self._prediction(next_latent_state)
        
        return MZNetworkOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            latent_state=next_latent_state
        )
    
    def _representation(self, observation):
        """Encode observation to latent state"""
        latent_state = self.representation_network(observation)
        if self.state_norm:
            latent_state = renormalize(latent_state)
        return latent_state
    
    def _afterstate_dynamics(self, latent_state, action):
        """Predict afterstate (deterministic dynamics)"""
        afterstate = self.afterstate_dynamics_network(latent_state, action)
        if self.state_norm:
            afterstate = renormalize(afterstate)
        return afterstate
    
    def _chance_encoding(self, afterstate, chance):
        """Apply chance to afterstate (stochastic dynamics)"""
        # Convert afterstate back to grid representation for chance encoding
        grid_repr = afterstate.view(-1, 16, self.grid_size, self.grid_size)
        next_state = self.chance_encoder(grid_repr, chance)
        if self.state_norm:
            next_state = renormalize(next_state)
        return next_state
    
    def _afterstate_reward(self, afterstate):
        """Predict reward from afterstate"""
        return self.reward_head_afterstate(afterstate)
    
    def _prediction(self, latent_state):
        """Predict value and policy"""
        value = self.value_head(latent_state)
        policy_logits = self.policy_head(latent_state)
        return value, policy_logits
    
    def get_params_mean(self):
        return get_params_mean(self)
    
    def get_dynamic_mean(self):
        return get_dynamic_mean(self)
    
    def get_reward_mean(self):
        return get_reward_mean(self)
