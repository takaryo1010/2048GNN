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
    - Edges connect: adjacent cells (up/down/left/right) + same row/column pairs
    - Node features: log2(tile_value), is_empty, position_encoding
    """
    
    def __init__(self, grid_size: int = 4, include_row_col_edges: bool = True):
        """
        Args:
            grid_size: Size of the square grid (default 4 for 4x4)
            include_row_col_edges: Whether to add edges for all pairs in same row/column
        """
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size
        self.include_row_col_edges = include_row_col_edges
        
        # Pre-compute edge indices (static for fixed grid)
        self.edge_index = self._build_edge_index()
    
    def _build_edge_index(self) -> torch.Tensor:
        """
        Build edge connectivity matrix.
        Returns edge_index of shape [2, num_edges]
        """
        edges = []
        
        # Add adjacent edges (4-connectivity: up, down, left, right)
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
        
        # Add row/column edges for long-range dependencies
        if self.include_row_col_edges:
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
        batch_size = obs.size(0)
        
        # Move edge_index to same device as obs
        edge_index = self.edge_index.to(obs.device)
        
        # Extract node features from observation
        # obs shape: [B, C, H, W] -> flatten spatial: [B, C, H*W] -> transpose: [B, H*W, C]
        node_features = obs.flatten(2).transpose(1, 2)  # [B, N, C]
        
        # Add positional encoding
        pos_encoding = self._get_positional_encoding(batch_size, obs.device)
        node_features = torch.cat([node_features, pos_encoding], dim=-1)
        
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
    GraphSAGE Convolution Layer
    Simple and robust GNN layer for aggregating neighbor information
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
        Forward pass
        
        Args:
            x: Node features [B, N, D_in] or [N, D_in]
            edge_index: Edge connectivity [2, E]
        
        Returns:
            out: Updated node features [B, N, D_out] or [N, D_out]
        """
        has_batch = x.dim() == 3
        
        if has_batch:
            batch_size, num_nodes, _ = x.size()
            device = x.device
            
            # Process each graph in batch separately
            outputs = []
            for b in range(batch_size):
                x_b = x[b]  # [N, D_in]
                out_b = self._forward_single(x_b, edge_index)
                outputs.append(out_b)
            
            out = torch.stack(outputs, dim=0)  # [B, N, D_out]
        else:
            out = self._forward_single(x, edge_index)
        
        return out
    
    def _forward_single(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for single graph (no batch dimension)
        
        Args:
            x: [N, D_in]
            edge_index: [2, E]
        """
        src, dst = edge_index[0], edge_index[1]
        
        # Aggregate neighbor features
        num_nodes = x.size(0)
        
        if self.agg == 'mean':
            # Count degree for each node
            deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
            deg = deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            deg = deg.clamp(min=1.0)
            
            # Sum neighbor features
            neigh = torch.zeros_like(x)
            neigh = neigh.index_add_(0, dst, x[src])
            
            # Average
            neigh = neigh / deg.unsqueeze(-1)
        
        elif self.agg == 'max':
            # Max pooling over neighbors
            neigh = torch.zeros_like(x).fill_(float('-inf'))
            neigh = torch.scatter_reduce(neigh, 0, dst.unsqueeze(-1).expand_as(x[src]), x[src], reduce='amax', include_self=False)
            neigh = neigh.clamp(min=0)  # Replace -inf with 0 for nodes without neighbors
        
        else:  # sum
            neigh = torch.zeros_like(x)
            neigh = neigh.index_add_(0, dst, x[src])
        
        # Concatenate self and neighbor features
        h = torch.cat([x, neigh], dim=-1)  # [N, 2*D_in]
        
        # Linear transformation + activation
        out = F.relu(self.lin(h))  # [N, D_out]
        
        return out


class GraphSAGE(nn.Module):
    """
    Multi-layer GraphSAGE network
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3, 
                 dropout: float = 0.0, use_bn: bool = True):
        """
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden feature dimension
            num_layers: Number of GraphSAGE layers
            dropout: Dropout probability
            use_bn: Whether to use batch normalization
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bn = use_bn
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_bn else None
        
        # First layer
        self.convs.append(GraphSAGEConv(in_dim, hidden_dim))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GraphSAGEConv(hidden_dim, hidden_dim))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
    
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
            
            if self.use_bn and self.bns is not None:
                # BatchNorm expects [B, C, ...] so we need to transpose
                # [B, N, D] -> [B, D, N] -> BatchNorm -> [B, N, D]
                x = x.transpose(1, 2)
                x = self.bns[i](x)
                x = x.transpose(1, 2)
            
            if i < self.num_layers - 1:  # No dropout on last layer
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
