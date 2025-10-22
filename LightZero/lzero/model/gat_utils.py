"""
Graph Attention Network (GAT) utilities for 2048 game
Uses multi-head attention mechanism for graph convolution
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GraphAttentionConv(nn.Module):
    """
    Graph Attention Convolution Layer (GAT)
    Implements multi-head attention mechanism for message passing
    
    Reference: "Graph Attention Networks" (Veličković et al., 2018)
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, 
                 concat: bool = True, dropout: float = 0.0, bias: bool = True):
        """
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension per head
            num_heads: Number of attention heads
            concat: If True, concatenate heads; if False, average them
            dropout: Dropout probability for attention coefficients
            bias: Whether to use bias in linear transformation
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        
        # Linear transformation for each head
        self.lin = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
        # Attention parameters (a in the paper)
        # For each head, we have 2 * out_dim parameters (for source and target)
        self.att = nn.Parameter(torch.Tensor(1, num_heads, 2 * out_dim))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(out_dim * num_heads))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-head attention
        
        Args:
            x: Node features [B, N, D_in]
            edge_index: Edge connectivity [2, E]
        
        Returns:
            out: Updated node features [B, N, D_out * num_heads] if concat
                 or [B, N, D_out] if average
        """
        batch_size, num_nodes, feat_dim = x.size()
        H = self.num_heads
        
        # Linear transformation: [B, N, D_in] -> [B, N, H * D_out]
        x_transformed = self.lin(x)
        
        # Reshape to separate heads: [B, N, H, D_out]
        x_transformed = x_transformed.view(batch_size, num_nodes, H, self.out_dim)
        
        # Compute attention coefficients for all edges
        # edge_index: [2, E] where E is number of edges
        src, dst = edge_index[0], edge_index[1]
        num_edges = src.size(0)
        
        # Prepare source and target features for attention
        # [B, E, H, D_out]
        x_src = x_transformed[:, src, :, :]  # [B, E, H, D_out]
        x_dst = x_transformed[:, dst, :, :]  # [B, E, H, D_out]
        
        # Concatenate source and target: [B, E, H, 2*D_out]
        x_edge = torch.cat([x_src, x_dst], dim=-1)
        
        # Compute attention logits: [B, E, H]
        # att: [1, H, 2*D_out]
        # x_edge: [B, E, H, 2*D_out]
        alpha = (x_edge * self.att).sum(dim=-1)  # [B, E, H]
        
        # Apply LeakyReLU
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        
        # Softmax per destination node (across incoming edges)
        # We need to group by destination node and apply softmax
        alpha_soft = self._edge_softmax(alpha, dst, num_nodes)  # [B, E, H]
        
        # Apply dropout to attention coefficients
        alpha_soft = F.dropout(alpha_soft, p=self.dropout, training=self.training)
        
        # Aggregate messages: weighted sum of source features
        # alpha_soft: [B, E, H] -> [B, E, H, 1]
        # x_src: [B, E, H, D_out]
        # out: [B, N, H, D_out]
        alpha_soft = alpha_soft.unsqueeze(-1)  # [B, E, H, 1]
        messages = alpha_soft * x_src  # [B, E, H, D_out]
        
        # Aggregate messages to destination nodes
        out = torch.zeros(batch_size, num_nodes, H, self.out_dim, 
                         dtype=x.dtype, device=x.device)
        
        # Expand dst for all dimensions: [E] -> [B, E, H, D_out]
        dst_expanded = dst.view(1, -1, 1, 1).expand(batch_size, num_edges, H, self.out_dim)
        
        # Scatter add: sum messages to destination nodes
        out = out.scatter_add(1, dst_expanded, messages)
        
        # Concatenate or average heads
        if self.concat:
            # [B, N, H, D_out] -> [B, N, H * D_out]
            out = out.reshape(batch_size, num_nodes, H * self.out_dim)
        else:
            # [B, N, H, D_out] -> [B, N, D_out]
            out = out.mean(dim=2)
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def _edge_softmax(self, alpha: torch.Tensor, dst: torch.Tensor, 
                     num_nodes: int) -> torch.Tensor:
        """
        Apply softmax to attention coefficients grouped by destination node
        
        Args:
            alpha: Attention logits [B, E, H]
            dst: Destination node indices [E]
            num_nodes: Total number of nodes
        
        Returns:
            alpha_soft: Softmax attention coefficients [B, E, H]
        """
        batch_size, num_edges, num_heads = alpha.size()
        
        # Compute max for numerical stability (per destination node)
        alpha_max = torch.full((batch_size, num_nodes, num_heads), 
                              float('-inf'), dtype=alpha.dtype, device=alpha.device)
        dst_expanded = dst.view(1, -1, 1).expand(batch_size, num_edges, num_heads)
        alpha_max = alpha_max.scatter_reduce(1, dst_expanded, alpha, 
                                             reduce='amax', include_self=False)
        alpha_max = alpha_max[:, dst, :]  # [B, E, H]
        
        # Subtract max and exponentiate
        alpha_exp = torch.exp(alpha - alpha_max)
        
        # Compute sum of exponentials per destination node
        alpha_sum = torch.zeros(batch_size, num_nodes, num_heads, 
                               dtype=alpha.dtype, device=alpha.device)
        alpha_sum = alpha_sum.scatter_add(1, dst_expanded, alpha_exp)
        alpha_sum = alpha_sum[:, dst, :].clamp(min=1e-16)  # [B, E, H]
        
        # Normalize
        alpha_soft = alpha_exp / alpha_sum
        
        return alpha_soft


class GraphAttention(nn.Module):
    """
    Multi-layer Graph Attention Network
    Stacks multiple GraphAttentionConv layers with residual connections
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.0, use_bn: bool = True):
        """
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden feature dimension (per head)
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_bn: Whether to use Layer Normalization
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_norm = use_bn
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_bn else None
        
        # First layer: in_dim -> hidden_dim * num_heads
        self.convs.append(
            GraphAttentionConv(in_dim, hidden_dim, num_heads=num_heads, 
                             concat=True, dropout=dropout)
        )
        if use_bn:
            self.norms.append(nn.LayerNorm(hidden_dim * num_heads))
        
        # Middle layers: hidden_dim * num_heads -> hidden_dim * num_heads
        for i in range(num_layers - 2):
            self.convs.append(
                GraphAttentionConv(hidden_dim * num_heads, hidden_dim, 
                                 num_heads=num_heads, concat=True, dropout=dropout)
            )
            if use_bn:
                self.norms.append(nn.LayerNorm(hidden_dim * num_heads))
        
        # Last layer: hidden_dim * num_heads -> hidden_dim (average heads)
        if num_layers > 1:
            self.convs.append(
                GraphAttentionConv(hidden_dim * num_heads, hidden_dim, 
                                 num_heads=num_heads, concat=False, dropout=dropout)
            )
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
            x_in = x
            x = conv(x, edge_index)
            
            # Apply normalization
            if self.use_norm and self.norms is not None:
                x = self.norms[i](x)
            
            # Apply activation (ReLU) except on last layer
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection (if dimensions match)
            if i > 0 and x_in.size(-1) == x.size(-1):
                x = x + x_in
        
        return x
