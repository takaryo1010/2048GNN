"""
Graph Neural Network utilities for 2048 game
Converts grid-based observations to graph structures (nodes, edges, features)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GraphBuilder:
    """
    Converts 2048 grid observations to graph structures.
    
    For a 4x4 grid (16 cells):
    - Each cell becomes a node
    - Edges connect: adjacent cells (up/down/left/right) + optionally same row/column pairs
    - Node features: log2(tile_value), is_empty, position_encoding
    """
    
    def __init__(self, grid_size: int = 4, include_row_col_edges: bool = True, 
                 edge_mode: str = 'full'):
        """
        Args:
            grid_size: Size of the square grid (default 4 for 4x4)
            include_row_col_edges: Whether to add edges for same row/column (backward compatibility)
            edge_mode: Edge connectivity mode:
                - 'adjacent': Only 4-connected neighbors (fastest, ~56 edges for 4x4)
                - 'sparse': Adjacent + next-nearest in rows/cols (fast, ~88 edges)
                - 'full': All pairs in same row/col (slow, ~200 edges)
        """
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size
        
        # Determine edge mode
        if not include_row_col_edges:
            self.edge_mode = 'adjacent'
        else:
            self.edge_mode = edge_mode
        
        # Pre-compute edge indices (static for fixed grid)
        self.edge_index = self._build_edge_index()
    
    def _build_edge_index(self) -> torch.Tensor:
        """
        Build edge connectivity matrix based on edge_mode.
        Returns edge_index of shape [2, num_edges]
        
        Edge counts for 4x4 grid:
        - adjacent: ~56 edges (4-connected only)
        - sparse: ~88 edges (4-connected + distance-2 in rows/cols)
        - full: ~200 edges (all pairs in same row/col)
        """
        edges = []
        
        # Always add adjacent edges (4-connectivity: up, down, left, right)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                node_id = i * self.grid_size + j
                
                # Right neighbor
                if j < self.grid_size - 1:
                    neighbor = i * self.grid_size + (j + 1)
                    edges.append([node_id, neighbor])
                    edges.append([neighbor, node_id])  # bidirectional
                
                # Down neighbor
                if i < self.grid_size - 1:
                    neighbor = (i + 1) * self.grid_size + j
                    edges.append([node_id, neighbor])
                    edges.append([neighbor, node_id])  # bidirectional
        
        # Add row/column edges based on mode
        if self.edge_mode == 'sparse':
            # Add edges to distance-2 neighbors in rows/cols (middle ground)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    node_id = i * self.grid_size + j
                    
                    # Distance-2 right
                    if j < self.grid_size - 2:
                        neighbor = i * self.grid_size + (j + 2)
                        edges.append([node_id, neighbor])
                        edges.append([neighbor, node_id])
                    
                    # Distance-2 down
                    if i < self.grid_size - 2:
                        neighbor = (i + 2) * self.grid_size + j
                        edges.append([node_id, neighbor])
                        edges.append([neighbor, node_id])
                        
        elif self.edge_mode == 'full':
            # Add all pairs in same row/column (original behavior)
            for i in range(self.grid_size):
                # Row edges: connect all cells in same row
                for j1 in range(self.grid_size):
                    for j2 in range(j1 + 1, self.grid_size):
                        node1 = i * self.grid_size + j1
                        node2 = i * self.grid_size + j2
                        edges.append([node1, node2])
                        edges.append([node2, node1])
                
                # Column edges: connect all cells in same column
                for i1 in range(self.grid_size):
                    for i2 in range(i1 + 1, self.grid_size):
                        node1 = i1 * self.grid_size + i
                        node2 = i2 * self.grid_size + i
                        edges.append([node1, node2])
                        edges.append([node2, node1])
        
        # edge_mode == 'adjacent': no additional edges
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def obs_to_graph(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert observation to graph structure.
        
        Args:
            obs: Observation tensor of shape [B, C, H, W] where C is number of channels
                 For 2048: typically [B, 16, 4, 4] (one-hot encoded)
        
        Returns:
            node_features: [B, N, D] where N=16 nodes, D is feature dimension
            edge_index: [2, E] where E is number of edges (shared across batch)
        """
        # バッチサイズ
        batch_size = obs.size(0)

        # edge_index を観測と同じデバイスに移す
        # - edge_index: [2, E], 事前計算されたエッジインデックス（グリッド接続）
        edge_index = self.edge_index.to(obs.device)

        # 観測からノード特徴を抽出
        # - obs: [B, C, H, W]
        # - obs.flatten(2) -> [B, C, H*W]
        # - transpose(1,2) -> [B, H*W, C]  (ここで N = H*W はノード数)
        node_features = obs.flatten(2).transpose(1, 2)  # [B, N, C]

        # 位置情報を追加
        # - pos_encoding: [B, N, 2] (row_norm, col_norm)
        # - concat -> node_features: [B, N, C+2]
        pos_encoding = self._get_positional_encoding(batch_size, obs.device)
        node_features = torch.cat([node_features, pos_encoding], dim=-1)

        # 戻り値:
        # - node_features: [B, N, D_in]  (D_in = C + 2)
        # - edge_index: [2, E]
        return node_features, edge_index
    
    def _get_positional_encoding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate positional encoding for each node (row_id, col_id normalized to [0,1])
        
        Returns:
            pos_encoding: [B, N, 2]
        """
        positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Normalize to [0, 1]
                row_norm = i / (self.grid_size - 1) if self.grid_size > 1 else 0.5
                col_norm = j / (self.grid_size - 1) if self.grid_size > 1 else 0.5
                positions.append([row_norm, col_norm])
        
        pos_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
        # Expand for batch: [N, 2] -> [B, N, 2]
        pos_encoding = pos_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        return pos_encoding


class GraphSAGEConv(nn.Module):
    """
    GraphSAGE Convolution Layer (Optimized for batched processing)
    Aggregates neighbor information efficiently across entire batch
    """
    
    def __init__(self, in_dim: int, out_dim: int, agg: str = 'mean', bias: bool = True):
        """
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            agg: Aggregation method ('mean', 'max', 'sum')
            bias: Whether to use bias in linear transformation
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.agg = agg
        
        # Linear transformation for concatenated [self_features, neighbor_features]
        self.lin = nn.Linear(in_dim * 2, out_dim, bias=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - optimized for batch processing
        
        Args:
            x: Node features [B, N, D_in]
            edge_index: Edge connectivity [2, E]
        
        Returns:
            out: Updated node features [B, N, D_out]
        """
        batch_size, num_nodes, feat_dim = x.size()
        src, dst = edge_index[0], edge_index[1]
        num_edges = src.size(0)
        
        # Flatten batch for efficient processing: [B, N, D] -> [B*N, D]
        x_flat = x.view(batch_size * num_nodes, feat_dim)
        
        # Create batch-aware edge indices
        # For each graph in batch, offset node indices by batch_idx * num_nodes
        edge_index_batch = []
        for b in range(batch_size):
            offset = b * num_nodes
            edge_index_batch.append(edge_index + offset)
        edge_index_batch = torch.cat(edge_index_batch, dim=1)  # [2, B*E]
        
        src_batch, dst_batch = edge_index_batch[0], edge_index_batch[1]
        
        # Aggregate neighbors for all batches at once
        if self.agg == 'mean':
            # Count degrees: [B*N]
            deg = torch.zeros(batch_size * num_nodes, device=x.device, dtype=x.dtype)
            deg = deg.index_add_(0, dst_batch, torch.ones_like(dst_batch, dtype=x.dtype))
            deg = deg.clamp(min=1.0)
            
            # Sum neighbor features: [B*N, D]
            neigh = torch.zeros_like(x_flat)
            neigh = neigh.index_add_(0, dst_batch, x_flat[src_batch])
            
            # Average
            neigh = neigh / deg.unsqueeze(-1)
            
        elif self.agg == 'sum':
            neigh = torch.zeros_like(x_flat)
            neigh = neigh.index_add_(0, dst_batch, x_flat[src_batch])
            
        else:  # max
            neigh = torch.zeros_like(x_flat).fill_(float('-inf'))
            neigh = torch.scatter_reduce(
                neigh, 0, 
                dst_batch.unsqueeze(-1).expand_as(x_flat[src_batch]), 
                x_flat[src_batch], 
                reduce='amax', 
                include_self=False
            )
            neigh = neigh.clamp(min=0)
        
        # Concatenate self and neighbor features: [B*N, 2*D]
        h = torch.cat([x_flat, neigh], dim=-1)
        
        # Linear transformation: [B*N, D_out]
        out = F.relu(self.lin(h))
        
        # Reshape back to batch format: [B, N, D_out]
        out = out.view(batch_size, num_nodes, self.out_dim)
        
        return out


class GraphSAGE(nn.Module):
    """
    Multi-layer GraphSAGE network (Optimized with LayerNorm)
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3, 
                 dropout: float = 0.0, use_bn: bool = True):
        """
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden feature dimension
            num_layers: Number of GraphSAGE layers
            dropout: Dropout probability
            use_bn: Whether to use normalization (uses LayerNorm for efficiency)
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_norm = use_bn  # Keep parameter name for compatibility
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_bn else None
        
        # First layer
        self.convs.append(GraphSAGEConv(in_dim, hidden_dim))
        if use_bn:
            # Use LayerNorm instead of BatchNorm - no transpose needed!
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GraphSAGEConv(hidden_dim, hidden_dim))
            if use_bn:
                self.norms.append(nn.LayerNorm(hidden_dim))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [B, N, D_in]
            edge_index: Edge connectivity [2, E]
        
        Returns:
            out: Updated node features [B, N, hidden_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.use_norm and self.norms is not None:
                # LayerNorm works directly on [B, N, D] - no transpose needed!
                x = self.norms[i](x)
            
            if i < self.num_layers - 1:  # No dropout on last layer
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
