"""
GAT-based Stochastic MuZero Model for 2048
Replaces CNN with Graph Attention Network (GAT) for state representation and dynamics

⚠️  CNN使用ポリシー ⚠️
==================
このモデルは完全にGraph Attention Network (GAT)ベースです。

【CNNの使用制限】
- ✅ 許可: chance_encoderのみ（チャンスノードエンコーディング用）
- ❌ 禁止: representation_network, dynamics_network, prediction_network
          でのCNN使用は完全に禁止

【使用されるGATコンポーネント】
- GraphBuilder: グリッド観測をグラフ構造に変換
- GraphAttention: グラフアテンションネットワーク（マルチヘッド）
- GraphAttentionConv: アテンションベースのメッセージパッシング層

【バリデーション】
初期化時に自動的にCNN使用チェックが実行され、
GAT部分でCNNが使用されている場合はRuntimeErrorが発生します。
"""
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.torch_utils import MLP
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import MZNetworkOutput
from .gnn_utils import GraphBuilder  # Reuse GraphBuilder
from .gat_utils import GraphAttention  # New GAT module
from .utils import renormalize


class GATRepresentationNetwork(nn.Module):
    """
    GAT-based Representation Network
    Converts observation to latent state using Graph Attention Network instead of CNN
    """
    
    def __init__(
        self,
        observation_shape: SequenceType = (16, 4, 4),
        num_channels: int = 128,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        grid_size: int = 4,
        include_row_col_edges: bool = True,
        dropout: float = 0.0,
        edge_mode: str = 'sparse',
    ):
        """
        Args:
            observation_shape: Shape of observation [C, H, W]
            num_channels: Hidden dimension per head for GAT
            num_gnn_layers: Number of GAT layers
            num_heads: Number of attention heads
            grid_size: Grid size (4 for 4x4)
            include_row_col_edges: Whether to include row/column edges
            dropout: Dropout rate
            edge_mode: Edge connectivity - 'adjacent', 'sparse', or 'full'
        """
        super().__init__()
        self.observation_shape = observation_shape
        self.num_channels = num_channels
        self.grid_size = grid_size
        
        # Graph builder with optimized edge mode (same as GNN)
        self.graph_builder = GraphBuilder(grid_size, include_row_col_edges, edge_mode)
        
        # Input dimension: observation channels + 2 (positional encoding)
        in_dim = observation_shape[0] + 2
        
        # GAT encoder with multi-head attention
        self.gat = GraphAttention(
            in_dim=in_dim,
            hidden_dim=num_channels,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_bn=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Observation [B, C, H, W]
        
        Returns:
            latent_state: [B, num_channels, H, W] to maintain compatibility with existing code
        """
        batch_size = x.size(0)
        
        # Convert observation to graph representation
        node_features, edge_index = self.graph_builder.obs_to_graph(x)
        
        # Apply GAT
        node_embeddings = self.gat(node_features, edge_index)
        
        # Reshape node embeddings to grid format
        latent_state = node_embeddings.transpose(1, 2).reshape(
            batch_size, self.num_channels, self.grid_size, self.grid_size
        )
        
        return latent_state


class GATValueHead(nn.Module):
    """
    GAT-based Value Head
    Aggregates node embeddings and predicts value
    """
    
    def __init__(
        self,
        num_channels: int,
        value_support_size: int,
        hidden_channels: SequenceType = [128, 64],
        last_linear_layer_init_zero: bool = True,
    ):
        super().__init__()
        self.num_channels = num_channels
        
        # Multiple aggregations: mean, max, sum
        aggregated_dim = num_channels * 3
        
        # MLP for value prediction
        layers = []
        dims = [aggregated_dim] + list(hidden_channels) + [value_support_size]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize last layer to zero if requested
        if last_linear_layer_init_zero:
            nn.init.constant_(self.mlp[-1].weight, 0)
            nn.init.constant_(self.mlp[-1].bias, 0)
    
    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_state: [B, C, H, W]
        
        Returns:
            value: [B, value_support_size]
        """
        batch_size = latent_state.size(0)
        
        # Flatten spatial dimensions: [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        node_emb = latent_state.flatten(2).transpose(1, 2)  # [B, N, C]
        
        # Aggregate across nodes
        mean_pool = node_emb.mean(dim=1)  # [B, C]
        max_pool = node_emb.max(dim=1)[0]  # [B, C]
        sum_pool = node_emb.sum(dim=1)  # [B, C]
        
        # Concatenate aggregations
        aggregated = torch.cat([mean_pool, max_pool, sum_pool], dim=-1)  # [B, 3*C]
        
        # Predict value
        value = self.mlp(aggregated)  # [B, value_support_size]
        
        return value


class GATPolicyHead(nn.Module):
    """
    GAT-based Policy Head
    Predicts action probabilities from node embeddings
    """
    
    def __init__(
        self,
        num_channels: int,
        action_space_size: int,
        hidden_channels: SequenceType = [128, 64],
        last_linear_layer_init_zero: bool = True,
    ):
        super().__init__()
        self.num_channels = num_channels
        
        # Multiple aggregations
        aggregated_dim = num_channels * 3
        
        # MLP for policy prediction
        layers = []
        dims = [aggregated_dim] + list(hidden_channels) + [action_space_size]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
        
        if last_linear_layer_init_zero:
            nn.init.constant_(self.mlp[-1].weight, 0)
            nn.init.constant_(self.mlp[-1].bias, 0)
    
    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_state: [B, C, H, W]
        
        Returns:
            policy_logits: [B, action_space_size]
        """
        # Flatten spatial: [B, C, H*W] -> [B, H*W, C]
        node_emb = latent_state.flatten(2).transpose(1, 2)
        
        # Aggregate
        mean_pool = node_emb.mean(dim=1)
        max_pool = node_emb.max(dim=1)[0]
        sum_pool = node_emb.sum(dim=1)
        aggregated = torch.cat([mean_pool, max_pool, sum_pool], dim=-1)
        
        # Predict policy
        policy_logits = self.mlp(aggregated)
        
        return policy_logits


class GATPredictionNetwork(nn.Module):
    """
    GAT-based Prediction Network
    Combines value and policy heads
    """
    
    def __init__(
        self,
        num_channels: int,
        action_space_size: int,
        value_support_size: int,
        value_head_hidden_channels: SequenceType = [128, 64],
        policy_head_hidden_channels: SequenceType = [128, 64],
        last_linear_layer_init_zero: bool = True,
    ):
        super().__init__()
        
        self.value_head = GATValueHead(
            num_channels,
            value_support_size,
            value_head_hidden_channels,
            last_linear_layer_init_zero
        )
        
        self.policy_head = GATPolicyHead(
            num_channels,
            action_space_size,
            policy_head_hidden_channels,
            last_linear_layer_init_zero
        )
    
    def forward(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_state: [B, C, H, W]
        
        Returns:
            policy_logits: [B, action_space_size]
            value: [B, value_support_size]
        """
        value = self.value_head(latent_state)
        policy_logits = self.policy_head(latent_state)
        
        return policy_logits, value


class GATDynamicsNetwork(nn.Module):
    """
    GAT-based Dynamics Network
    Predicts next latent state and reward given current state and action
    """
    
    def __init__(
        self,
        num_channels: int,
        action_space_size: int,
        reward_support_size: int,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        grid_size: int = 4,
        reward_head_hidden_channels: SequenceType = [128, 64],
        last_linear_layer_init_zero: bool = True,
        include_row_col_edges: bool = True,
        edge_mode: str = 'sparse',
    ):
        super().__init__()
        self.num_channels = num_channels
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size
        
        # Graph builder with optimized edge mode
        self.graph_builder = GraphBuilder(grid_size, include_row_col_edges, edge_mode)
        
        # Action encoding: broadcast action to all nodes
        self.action_encoder = nn.Linear(action_space_size, num_channels)
        
        # GAT to predict next state
        self.gat = GraphAttention(
            in_dim=num_channels * 2,  # state + encoded action
            hidden_dim=num_channels,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=0.0,
            use_bn=True
        )
        
        # Reward head (similar to value head)
        aggregated_dim = num_channels * 3
        layers = []
        dims = [aggregated_dim] + list(reward_head_hidden_channels) + [reward_support_size]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.reward_head = nn.Sequential(*layers)
        
        if last_linear_layer_init_zero:
            nn.init.constant_(self.reward_head[-1].weight, 0)
            nn.init.constant_(self.reward_head[-1].bias, 0)
    
    def forward(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_state: [B, C, H, W]
            action: [B, A] one-hot encoded action
        
        Returns:
            next_latent_state: [B, C, H, W]
            reward: [B, reward_support_size]
        """
        batch_size = latent_state.size(0)
        
        # Convert latent state to node features: [B, C, H, W] -> [B, H*W, C]
        node_features = latent_state.flatten(2).transpose(1, 2)  # [B, N, C]
        
        # Encode action and broadcast to all nodes
        # Handle both one-hot encoded and index-based actions
        if action.dim() == 1 or (action.dim() == 2 and action.size(1) == 1):
            # Action is index-based, convert to one-hot
            if action.dim() == 2:
                action = action.squeeze(-1)
            # Determine action space size from encoder input size
            action_space_size = self.action_encoder.in_features
            # Clamp action indices to valid range
            action = action.long().clamp(0, action_space_size - 1)
            action = F.one_hot(action, num_classes=action_space_size).float()
        elif action.dtype != torch.float32:
            action = action.float()
        action_emb = self.action_encoder(action)  # [B, C]
        action_emb = action_emb.unsqueeze(1).expand(-1, self.num_nodes, -1)  # [B, N, C]
        
        # Concatenate state and action
        node_features = torch.cat([node_features, action_emb], dim=-1)  # [B, N, 2*C]
        
        # Get edge index
        edge_index = self.graph_builder.edge_index.to(latent_state.device)
        
        # Apply GAT to predict next state
        next_node_features = self.gat(node_features, edge_index)  # [B, N, C]
        
        # Reshape back to grid
        next_latent_state = next_node_features.transpose(1, 2).reshape(
            batch_size, self.num_channels, self.grid_size, self.grid_size
        )
        
        # Predict reward using aggregated features
        mean_pool = next_node_features.mean(dim=1)
        max_pool = next_node_features.max(dim=1)[0]
        sum_pool = next_node_features.sum(dim=1)
        aggregated = torch.cat([mean_pool, max_pool, sum_pool], dim=-1)
        reward = self.reward_head(aggregated)
        
        return next_latent_state, reward


@MODEL_REGISTRY.register('GATStochasticMuZeroModel')
class GATStochasticMuZeroModel(nn.Module):
    """
    GAT-based Stochastic MuZero Model for 2048
    Replaces CNN-based components with GAT (Graph Attention Network)
    """
    
    def __init__(
        self,
        observation_shape: SequenceType = (16, 4, 4),
        action_space_size: int = 4,
        chance_space_size: int = 32,
        num_channels: int = 128,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        grid_size: int = 4,
        value_head_hidden_channels: SequenceType = [128, 64],
        policy_head_hidden_channels: SequenceType = [128, 64],
        reward_head_hidden_channels: SequenceType = [128, 64],
        reward_support_size: int = 601,
        value_support_size: int = 601,
        categorical_distribution: bool = True,
        last_linear_layer_init_zero: bool = True,
        include_row_col_edges: bool = True,
        dropout: float = 0.0,
        # SSL parameters (if needed in future)
        self_supervised_learning_loss: bool = False,
        proj_hid: int = 1024,
        proj_out: int = 1024,
        pred_hid: int = 512,
        pred_out: int = 1024,
        *args,
        **kwargs
    ):
        """
        Args:
            observation_shape: Observation shape [C, H, W]
            action_space_size: Number of actions (4 for 2048)
            chance_space_size: Number of chance outcomes
            num_channels: Hidden dimension per head for GAT
            num_gnn_layers: Number of GAT layers
            num_heads: Number of attention heads
            grid_size: Grid size (4 for 4x4)
            value_head_hidden_channels: Hidden layers for value head
            policy_head_hidden_channels: Hidden layers for policy head
            reward_head_hidden_channels: Hidden layers for reward head
            reward_support_size: Support size for categorical reward
            value_support_size: Support size for categorical value
            categorical_distribution: Whether to use categorical distribution
            last_linear_layer_init_zero: Zero init for last layer
            include_row_col_edges: Whether to include row/column edges in graph
            dropout: Dropout rate
            self_supervised_learning_loss: Whether to use SSL (future feature)
        """
        super().__init__()
        
        self.observation_shape = observation_shape
        self.action_space_size = action_space_size
        self.chance_space_size = chance_space_size
        self.num_channels = num_channels
        self.categorical_distribution = categorical_distribution
        self.self_supervised_learning_loss = self_supervised_learning_loss
        
        if categorical_distribution:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size
        else:
            self.reward_support_size = 1
            self.value_support_size = 1
        
        # Determine optimal edge mode (sparse for good balance of speed/accuracy)
        edge_mode = kwargs.get('edge_mode', 'sparse')
        
        # Representation Network (GAT-based)
        self.representation_network = GATRepresentationNetwork(
            observation_shape=observation_shape,
            num_channels=num_channels,
            num_gnn_layers=num_gnn_layers,
            num_heads=num_heads,
            grid_size=grid_size,
            include_row_col_edges=include_row_col_edges,
            dropout=dropout,
            edge_mode=edge_mode,
        )
        
        # Prediction Network (GAT-based)
        self.prediction_network = GATPredictionNetwork(
            num_channels=num_channels,
            action_space_size=action_space_size,
            value_support_size=self.value_support_size,
            value_head_hidden_channels=value_head_hidden_channels,
            policy_head_hidden_channels=policy_head_hidden_channels,
            last_linear_layer_init_zero=last_linear_layer_init_zero,
        )
        
        # Dynamics Network (GAT-based)
        self.dynamics_network = GATDynamicsNetwork(
            num_channels=num_channels,
            action_space_size=action_space_size,
            reward_support_size=self.reward_support_size,
            num_gnn_layers=num_gnn_layers,
            num_heads=num_heads,
            grid_size=grid_size,
            edge_mode=edge_mode,
            reward_head_hidden_channels=reward_head_hidden_channels,
            last_linear_layer_init_zero=last_linear_layer_init_zero,
            include_row_col_edges=include_row_col_edges,
        )
        
        # Afterstate networks (for Stochastic MuZero)
        self.afterstate_dynamics_network = GATDynamicsNetwork(
            num_channels=num_channels,
            action_space_size=chance_space_size,  # Use chance space for afterstate
            reward_support_size=self.reward_support_size,
            num_gnn_layers=num_gnn_layers,
            num_heads=num_heads,
            grid_size=grid_size,
            reward_head_hidden_channels=reward_head_hidden_channels,
            last_linear_layer_init_zero=last_linear_layer_init_zero,
            include_row_col_edges=include_row_col_edges,
        )
        
        self.afterstate_prediction_network = GATPredictionNetwork(
            num_channels=num_channels,
            action_space_size=chance_space_size,  # Predict chance distribution
            value_support_size=self.value_support_size,
            value_head_hidden_channels=value_head_hidden_channels,
            policy_head_hidden_channels=policy_head_hidden_channels,
            last_linear_layer_init_zero=last_linear_layer_init_zero,
        )
        
        # Chance encoder - needs to match ChanceEncoder interface from stochastic_muzero_model.py
        from .stochastic_muzero_model import ChanceEncoder
        self.chance_encoder = ChanceEncoder(
            observation_shape, chance_space_size, encoder_backbone_type='conv'
        )
        
        # CNN使用を防止するバリデーション
        self._validate_no_cnn_in_gat_components()
    
    def _validate_no_cnn_in_gat_components(self):
        """
        GAT部分（representation, dynamics）にCNNが使われていないことを確認
        chance_encoderのCNNは除外（チャンスノード用として許可）
        
        このメソッドは初期化時に呼ばれ、GATモデルが誤ってCNNコンポーネントを
        使用しないことを保証します。
        """
        prohibited_cnn_types = ['Conv2d', 'ResBlock', 'BatchNorm2d']
        
        for name, module in self.named_modules():
            module_type = type(module).__name__
            
            # chance_encoder以外でCNNレイヤーを検出
            if 'chance_encoder' not in name:
                if any(cnn_type in module_type for cnn_type in prohibited_cnn_types):
                    raise RuntimeError(
                        f"❌ GAT部分でCNNレイヤーが検出されました！\n"
                        f"   検出場所: {name}\n"
                        f"   レイヤータイプ: {module_type}\n"
                        f"   このモデルはGraph Attention Network (GAT)ベースです。\n"
                        f"   CNNレイヤー（Conv2d, ResBlock, BatchNorm2d）の使用は禁止されています。\n"
                        f"   例外: chance_encoderのみCNNが許可されています。"
                    )
        
        # GATコンポーネントの存在確認
        has_graphattention = False
        has_gat_repr = False
        has_gat_dyn = False
        
        for name, module in self.named_modules():
            module_type = type(module).__name__
            if 'GraphAttention' in module_type:
                has_graphattention = True
            if 'GATRepresentationNetwork' in module_type:
                has_gat_repr = True
            if 'GATDynamicsNetwork' in module_type:
                has_gat_dyn = True  # Fixed: was False
        
        if not (has_graphattention and has_gat_repr and has_gat_dyn):
            raise RuntimeError(
                f"❌ 必須GATコンポーネントが見つかりません！\n"
                f"   GraphAttention: {'✅' if has_graphattention else '❌'}\n"
                f"   GATRepresentationNetwork: {'✅' if has_gat_repr else '❌'}\n"
                f"   GATDynamicsNetwork: {'✅' if has_gat_dyn else '❌'}\n"
                f"   このモデルは完全にGATベースである必要があります。"
            )
    
    def chance_encode(self, observation: torch.Tensor):
        """
        Encode observation to chance outcome distribution
        
        Args:
            observation: [B, C, H, W]
        
        Returns:
            chance_encoding: [B, chance_space_size]
            chance_onehot: [B, chance_space_size] one-hot encoded
        """
        output = self.chance_encoder(observation)
        return output
    
    def initial_inference(self, obs: torch.Tensor) -> MZNetworkOutput:
        """
        Initial inference: obs -> latent_state -> value, policy
        
        Args:
            obs: [B, C, H, W]
        
        Returns:
            MZNetworkOutput with value, reward, policy_logits, latent_state
        """
        batch_size = obs.size(0)
        latent_state = self._representation(obs)
        policy_logits, value = self._prediction(latent_state)
        
        return MZNetworkOutput(
            value=value,
            reward=[0.0 for _ in range(batch_size)],
            policy_logits=policy_logits,
            latent_state=latent_state,
        )
    
    def recurrent_inference(
        self,
        state: torch.Tensor,
        option: torch.Tensor,
        afterstate: bool = False
    ) -> MZNetworkOutput:
        """
        Recurrent inference: state + option -> next_state, reward, value, policy
        
        Args:
            state: [B, C, H, W] - latent_state or afterstate
            option: [B] or [B, A] - action or chance
            afterstate: Whether current state is afterstate
        
        Returns:
            MZNetworkOutput
        
        Notes:
            - afterstate=False: state is latent_state, option is action
              -> use dynamics_network to get afterstate, then afterstate_prediction_network
              -> policy_logits has chance_space_size dimensions
            - afterstate=True: state is afterstate, option is chance
              -> use afterstate_dynamics_network to get next_latent_state, then prediction_network
              -> policy_logits has action_space_size dimensions
        """
        if afterstate:
            # state is afterstate, option is chance
            # afterstate + chance -> next_latent_state
            next_latent_state, reward = self.afterstate_dynamics_network(state, option)
            # predict action distribution from next_latent_state
            policy_logits, value = self.prediction_network(next_latent_state)
            return MZNetworkOutput(value, reward, policy_logits, next_latent_state)
        else:
            # state is latent_state, option is action
            # latent_state + action -> afterstate
            next_afterstate, reward = self.dynamics_network(state, option)
            # predict chance distribution from afterstate
            policy_logits, value = self.afterstate_prediction_network(next_afterstate)
            return MZNetworkOutput(value, reward, policy_logits, next_afterstate)
    
    def _representation(self, obs: torch.Tensor) -> torch.Tensor:
        """Representation network forward"""
        return self.representation_network(obs)
    
    def _prediction(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prediction network forward"""
        return self.prediction_network(latent_state)
    
    def _dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dynamics network forward"""
        return self.dynamics_network(latent_state, action)
    
    def _afterstate_dynamics(self, afterstate: torch.Tensor, chance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Afterstate dynamics network forward"""
        return self.afterstate_dynamics_network(afterstate, chance)
    
    def _afterstate_prediction(self, afterstate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Afterstate prediction network forward"""
        return self.afterstate_prediction_network(afterstate)
    
    def project(self, latent_state: torch.Tensor, with_grad: bool = True):
        """
        Project latent state for self-supervised learning
        (Placeholder for future SSL implementation)
        """
        # Use reshape instead of view for better compatibility with non-contiguous tensors
        latent_state = latent_state.reshape(latent_state.size(0), -1)
        
        # Simple projection
        proj = latent_state
        
        if with_grad:
            proj = proj / (torch.norm(proj, dim=-1, keepdim=True) + 1e-8)
            return proj
        else:
            with torch.no_grad():
                proj = proj / (torch.norm(proj, dim=-1, keepdim=True) + 1e-8)
                return proj
