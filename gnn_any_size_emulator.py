"""
GNN 2048 汎用サイズエミュレータ

4×4で学習したGNNモデルを任意の盤面サイズで動作させる独立したエミュレータです。
LightZeroのGUIに依存せず、コマンドラインで簡単に実行できます。

使い方:
    python gnn_any_size_emulator.py --grid-size 3 --episodes 10
    python gnn_any_size_emulator.py --grid-size 5 --episodes 5 --render
    python gnn_any_size_emulator.py --grid-size 6 --model-path path/to/model.pth.tar

主な機能:
    - 3×3から8×8まで任意のサイズの盤面でプレイ可能
    - リアルタイム描画モード（--render）
    - GIFアニメーション保存（--save-gif）
    - 詳細な統計情報の表示
    - カスタムモデルパスの指定
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import time

# LightZeroのパスを追加
sys.path.append('./LightZero')


# =============================================================================
# ゲーム環境（任意サイズ対応版）
# =============================================================================

class Game2048AnySize:
    """任意のサイズの2048ゲーム環境"""
    
    def __init__(self, grid_size: int = 4):
        """
        Args:
            grid_size: 盤面のサイズ（3〜8を推奨）
        """
        self.grid_size = grid_size
        self.board = None
        self.score = 0
        self.max_tile = 0
        self.moves = 0
        self.reset()
    
    def reset(self):
        """ゲームをリセット"""
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.score = 0
        self.max_tile = 0
        self.moves = 0
        
        # 初期タイルを2つ配置
        self._add_random_tile()
        self._add_random_tile()
        
        return self._get_observation()
    
    def _add_random_tile(self):
        """ランダムな空きマスに新しいタイルを追加"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = empty_cells[np.random.randint(len(empty_cells))]
            # 90%の確率で2、10%の確率で4
            self.board[row, col] = 2 if np.random.random() < 0.9 else 4
    
    def _get_observation(self):
        """観測を取得（ワンホットエンコーディング形式）"""
        # 0から2048までのlog2値をエンコード（16チャンネル）
        obs = np.zeros((16, self.grid_size, self.grid_size), dtype=np.float32)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.board[i, j] > 0:
                    # log2(value)をチャンネルインデックスとして使用
                    channel = min(int(np.log2(self.board[i, j])), 15)
                    obs[channel, i, j] = 1.0
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        アクションを実行
        
        Args:
            action: 0=上, 1=右, 2=下, 3=左
        
        Returns:
            observation: 新しい観測
            reward: 報酬
            done: 終了フラグ
            info: 追加情報
        """
        old_board = self.board.copy()
        old_score = self.score
        
        # アクションを実行
        moved = self._move(action)
        
        # 移動が発生した場合、新しいタイルを追加
        if moved:
            self._add_random_tile()
            self.moves += 1
        
        # 報酬計算
        reward = self.score - old_score
        
        # 最大タイルを更新
        self.max_tile = np.max(self.board)
        
        # 終了判定
        done = not self._has_legal_moves()
        
        info = {
            'score': self.score,
            'max_tile': self.max_tile,
            'moves': self.moves,
            'legal_move': moved
        }
        
        return self._get_observation(), reward, done, info
    
    def _move(self, action: int) -> bool:
        """
        盤面を移動させる
        
        Returns:
            moved: 移動が発生したかどうか
        """
        old_board = self.board.copy()
        
        if action == 0:  # 上
            self._move_up()
        elif action == 1:  # 右
            self._move_right()
        elif action == 2:  # 下
            self._move_down()
        elif action == 3:  # 左
            self._move_left()
        
        return not np.array_equal(old_board, self.board)
    
    def _move_left(self):
        """左に移動"""
        for i in range(self.grid_size):
            self.board[i, :] = self._merge_line(self.board[i, :])
    
    def _move_right(self):
        """右に移動"""
        for i in range(self.grid_size):
            self.board[i, :] = self._merge_line(self.board[i, ::-1])[::-1]
    
    def _move_up(self):
        """上に移動"""
        self.board = self.board.T
        self._move_left()
        self.board = self.board.T
    
    def _move_down(self):
        """下に移動"""
        self.board = self.board.T
        self._move_right()
        self.board = self.board.T
    
    def _merge_line(self, line: np.ndarray) -> np.ndarray:
        """1行をマージ"""
        # 0を除去
        non_zero = line[line != 0]
        
        if len(non_zero) == 0:
            return line
        
        # マージ
        merged = []
        skip = False
        
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                # マージ
                merged_value = non_zero[i] * 2
                merged.append(merged_value)
                self.score += merged_value
                skip = True
            else:
                merged.append(non_zero[i])
        
        # 0で埋める
        result = np.zeros(self.grid_size, dtype=np.int32)
        result[:len(merged)] = merged
        
        return result
    
    def _has_legal_moves(self) -> bool:
        """合法手があるかチェック"""
        # 空きマスがある
        if np.any(self.board == 0):
            return True
        
        # 隣接するマスに同じ値がある
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                current = self.board[i, j]
                # 右
                if j + 1 < self.grid_size and self.board[i, j + 1] == current:
                    return True
                # 下
                if i + 1 < self.grid_size and self.board[i + 1, j] == current:
                    return True
        
        return False
    
    def get_legal_actions(self) -> List[int]:
        """合法なアクションのリストを取得"""
        legal_actions = []
        
        for action in range(4):
            old_board = self.board.copy()
            self._move(action)
            if not np.array_equal(old_board, self.board):
                legal_actions.append(action)
            self.board = old_board
        
        return legal_actions


# =============================================================================
# GNN モデル（任意サイズ対応版）
# =============================================================================

class GraphBuilder:
    """グラフ構造を構築（任意サイズ対応）"""
    
    def __init__(self, grid_size: int, include_row_col_edges: bool = True, edge_mode: str = 'sparse'):
        self.grid_size = grid_size
        self.include_row_col_edges = include_row_col_edges
        self.edge_mode = edge_mode
        self.edge_index = None
    
    def obs_to_graph(self, obs: torch.Tensor):
        """観測をグラフ表現に変換"""
        batch_size = obs.size(0)
        num_nodes = self.grid_size * self.grid_size
        
        # ノード特徴量: [B, C, H, W] -> [B, N, C]
        node_features = obs.flatten(2).transpose(1, 2)  # [B, N, C]
        
        # 位置エンコーディングを追加
        pos_encoding = self._get_positional_encoding(batch_size, obs.device)
        node_features = torch.cat([node_features, pos_encoding], dim=-1)
        
        # エッジインデックスを取得（初回のみ計算）
        if self.edge_index is None:
            self.edge_index = self._build_edge_index(obs.device)
        
        return node_features, self.edge_index
    
    def _get_positional_encoding(self, batch_size: int, device: torch.device):
        """位置エンコーディングを取得"""
        pos_x = torch.arange(self.grid_size, device=device).float() / self.grid_size
        pos_y = torch.arange(self.grid_size, device=device).float() / self.grid_size
        
        pos_grid_x, pos_grid_y = torch.meshgrid(pos_x, pos_y, indexing='ij')
        pos_encoding = torch.stack([pos_grid_x, pos_grid_y], dim=-1)  # [H, W, 2]
        pos_encoding = pos_encoding.reshape(-1, 2)  # [N, 2]
        pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, 2]
        
        return pos_encoding
    
    def _build_edge_index(self, device: torch.device):
        """エッジインデックスを構築"""
        edges = []
        num_nodes = self.grid_size * self.grid_size
        
        # 4近傍エッジ（必須）
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                node_id = i * self.grid_size + j
                
                # 右
                if j + 1 < self.grid_size:
                    neighbor_id = i * self.grid_size + (j + 1)
                    edges.append([node_id, neighbor_id])
                    edges.append([neighbor_id, node_id])
                
                # 下
                if i + 1 < self.grid_size:
                    neighbor_id = (i + 1) * self.grid_size + j
                    edges.append([node_id, neighbor_id])
                    edges.append([neighbor_id, node_id])
        
        # sparse/fullモードの場合、追加のエッジを追加
        if self.edge_mode in ['sparse', 'full'] and self.include_row_col_edges:
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    node_id = i * self.grid_size + j
                    
                    if self.edge_mode == 'sparse':
                        # 距離2のエッジ
                        if j + 2 < self.grid_size:
                            neighbor_id = i * self.grid_size + (j + 2)
                            edges.append([node_id, neighbor_id])
                            edges.append([neighbor_id, node_id])
                        
                        if i + 2 < self.grid_size:
                            neighbor_id = (i + 2) * self.grid_size + j
                            edges.append([node_id, neighbor_id])
                            edges.append([neighbor_id, node_id])
                    
                    elif self.edge_mode == 'full':
                        # 同じ行・列の全ノードとエッジ
                        for k in range(self.grid_size):
                            if k != j:  # 同じ行
                                neighbor_id = i * self.grid_size + k
                                edges.append([node_id, neighbor_id])
                            
                            if k != i:  # 同じ列
                                neighbor_id = k * self.grid_size + j
                                edges.append([node_id, neighbor_id])
        
        # セルフループ
        for node_id in range(num_nodes):
            edges.append([node_id, node_id])
        
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
        
        return edge_index


class GraphSAGE(nn.Module):
    """GraphSAGEレイヤー（任意サイズ対応）"""
    
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.0, use_bn: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_bn else None
        
        # 入力層
        self.convs.append(self._make_conv(in_dim, hidden_dim))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # 隠れ層
        for _ in range(num_layers - 1):
            self.convs.append(self._make_conv(hidden_dim, hidden_dim))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
    
    def _make_conv(self, in_dim: int, out_dim: int):
        """SAGEConvレイヤーを作成（簡易版）"""
        return nn.Linear(in_dim * 2, out_dim)  # 自分とneighborの結合
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Args:
            x: [B, N, D] ノード特徴量
            edge_index: [2, E] エッジインデックス
        
        Returns:
            x: [B, N, H] 更新されたノード特徴量
        """
        batch_size, num_nodes, _ = x.size()
        
        for i, conv in enumerate(self.convs):
            # メッセージパッシング
            x = self._message_passing(x, edge_index, conv)
            
            # Batch Normalization
            if self.bns is not None:
                # [B, N, H] -> [B, H, N] -> BN -> [B, N, H]
                x = x.transpose(1, 2)
                x = self.bns[i](x)
                x = x.transpose(1, 2)
            
            # Activation
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def _message_passing(self, x: torch.Tensor, edge_index: torch.Tensor, conv: nn.Module):
        """メッセージパッシング（最適化版：ベクトル演算を使用）"""
        batch_size, num_nodes, feat_dim = x.size()
        
        # エッジインデックスから隣接ノードを取得
        src_nodes = edge_index[0]  # [E]
        dst_nodes = edge_index[1]  # [E]
        
        # ソースノードの特徴量を取得 [B, E, D]
        # x[:, src_nodes, :] で全バッチのソース特徴量を一度に取得
        src_features = x[:, src_nodes, :]  # [B, E, D]
        
        # 宛先ノードごとに集約（平均を計算）
        # scatter_add を使って効率的に集約
        aggregated = torch.zeros(batch_size, num_nodes, feat_dim, device=x.device)
        count = torch.zeros(batch_size, num_nodes, 1, device=x.device)
        
        # dst_nodesを[B, E, D]の形状に合わせて拡張
        dst_nodes_expanded = dst_nodes.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, feat_dim)
        
        # scatter_add で集約（合計を計算）
        aggregated = aggregated.scatter_add(1, dst_nodes_expanded, src_features)
        
        # 各ノードへのエッジ数をカウント
        count_expanded = dst_nodes.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        count = count.scatter_add(1, count_expanded, torch.ones_like(src_features[:, :, :1]))
        
        # 平均を計算（0除算を避ける）
        aggregated = aggregated / (count + 1e-8)
        
        # 自分の特徴量と隣接の特徴量を結合
        combined = torch.cat([x, aggregated], dim=-1)  # [B, N, 2*D]
        
        # 線形変換
        output = conv(combined)  # [B, N, H]
        
        return output


class GNNValueHead(nn.Module):
    """GNNバリューヘッド（任意サイズ対応）"""
    
    def __init__(
        self,
        num_channels: int,
        value_support_size: int,
        hidden_channels: List[int] = [128, 64],
    ):
        super().__init__()
        self.num_channels = num_channels
        
        aggregated_dim = num_channels * 3
        
        layers = []
        dims = [aggregated_dim] + list(hidden_channels) + [value_support_size]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_state: [B, C, H, W]
        
        Returns:
            value: [B, value_support_size]
        """
        node_emb = latent_state.flatten(2).transpose(1, 2)  # [B, N, C]
        
        mean_pool = node_emb.mean(dim=1)
        max_pool = node_emb.max(dim=1)[0]
        sum_pool = node_emb.sum(dim=1)
        aggregated = torch.cat([mean_pool, max_pool, sum_pool], dim=-1)
        
        value = self.mlp(aggregated)
        
        return value


class GNNRepresentationNetwork(nn.Module):
    """GNN表現ネットワーク（任意サイズ対応）"""
    
    def __init__(
        self,
        observation_shape: Tuple[int, int, int] = (16, 4, 4),
        num_channels: int = 128,
        num_gnn_layers: int = 3,
        grid_size: int = 4,
        include_row_col_edges: bool = True,
        dropout: float = 0.0,
        edge_mode: str = 'sparse',
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.num_channels = num_channels
        self.grid_size = grid_size
        
        self.graph_builder = GraphBuilder(grid_size, include_row_col_edges, edge_mode)
        
        in_dim = observation_shape[0] + 2  # obs_channels + positional encoding
        
        self.gnn = GraphSAGE(
            in_dim=in_dim,
            hidden_dim=num_channels,
            num_layers=num_gnn_layers,
            dropout=dropout,
            use_bn=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        
        Returns:
            latent_state: [B, num_channels, H, W]
        """
        batch_size = x.size(0)
        
        node_features, edge_index = self.graph_builder.obs_to_graph(x)
        node_embeddings = self.gnn(node_features, edge_index)
        
        latent_state = node_embeddings.transpose(1, 2).reshape(
            batch_size, self.num_channels, self.grid_size, self.grid_size
        )
        
        return latent_state


class GNNPolicyHead(nn.Module):
    """GNNポリシーヘッド（任意サイズ対応）"""
    
    def __init__(
        self,
        num_channels: int,
        action_space_size: int,
        hidden_channels: List[int] = [128, 64],
    ):
        super().__init__()
        self.num_channels = num_channels
        
        aggregated_dim = num_channels * 3
        
        layers = []
        dims = [aggregated_dim] + list(hidden_channels) + [action_space_size]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_state: [B, C, H, W]
        
        Returns:
            policy_logits: [B, action_space_size]
        """
        node_emb = latent_state.flatten(2).transpose(1, 2)  # [B, N, C]
        
        mean_pool = node_emb.mean(dim=1)
        max_pool = node_emb.max(dim=1)[0]
        sum_pool = node_emb.sum(dim=1)
        aggregated = torch.cat([mean_pool, max_pool, sum_pool], dim=-1)
        
        policy_logits = self.mlp(aggregated)
        
        return policy_logits


class GNNAgent:
    """GNNエージェント（任意サイズ対応）"""
    
    def __init__(
        self,
        model_path: str,
        grid_size: int = 4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_mcts: bool = False,
        num_simulations: int = 50
    ):
        """
        Args:
            model_path: 学習済みモデルのパス
            grid_size: 盤面のサイズ
            device: デバイス
            use_mcts: MCTSを使用するかどうか
            num_simulations: MCTSのシミュレーション回数
        """
        self.grid_size = grid_size
        self.device = device
        self.use_mcts = use_mcts
        self.num_simulations = num_simulations
        self.num_channels = 128  # デフォルト値、モデルロード時に調整される可能性あり
        self.num_gnn_layers = 3
        
        # モデルを構築
        self._build_model()
        
        # モデルをロード（4×4で学習したモデルを任意サイズに適用）
        self._load_model(model_path)
        
        # 評価モード
        self.representation_net.eval()
        self.policy_head.eval()
        if use_mcts:
            self.value_head.eval()
            print(f"✓ MCTSモード有効（シミュレーション回数: {num_simulations}）")
    
    def _build_model(self):
        """モデルを構築（または再構築）"""
        self.representation_net = GNNRepresentationNetwork(
            observation_shape=(16, self.grid_size, self.grid_size),
            num_channels=self.num_channels,
            num_gnn_layers=self.num_gnn_layers,
            grid_size=self.grid_size,
            include_row_col_edges=True,
            dropout=0.0,
            edge_mode='sparse'
        ).to(self.device)
        
        self.policy_head = GNNPolicyHead(
            num_channels=self.num_channels,
            action_space_size=4,
            hidden_channels=[128, 64]
        ).to(self.device)
        
        # MCTSを使用する場合、value headも必要
        if self.use_mcts:
            self.value_head = GNNValueHead(
                num_channels=self.num_channels,
                value_support_size=601,  # LightZeroのデフォルト
                hidden_channels=[128, 64]
            ).to(self.device)
    
    def _load_model(self, model_path: str):
        """学習済みモデルをロード"""
        print(f"モデルをロード中: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # state_dictを取得
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # チェックポイントからnum_channelsを推測
        detected_num_channels = None
        for key, value in state_dict.items():
            if 'representation_network.gnn.convs.0.lin.weight' in key:
                # GNN 1層目の出力次元からnum_channelsを推測
                detected_num_channels = value.shape[0]
                break
            elif 'gnn.convs.0.lin.weight' in key:
                detected_num_channels = value.shape[0]
                break
        
        # num_channelsが異なる場合は再構築
        if detected_num_channels is not None and detected_num_channels != self.num_channels:
            print(f"警告: モデルのnum_channelsが異なります（現在: {self.num_channels}, チェックポイント: {detected_num_channels}）")
            print(f"モデルを再構築します...")
            self.num_channels = detected_num_channels
            self._build_model()
        
        # キーの変換（必要に応じて）
        rep_state_dict = {}
        policy_state_dict = {}
        value_state_dict = {}
        
        for key, value in state_dict.items():
            if 'representation_network' in key:
                new_key = key.replace('representation_network.', '')
                rep_state_dict[new_key] = value
            elif 'prediction_network.policy_head' in key:
                new_key = key.replace('prediction_network.policy_head.', '')
                policy_state_dict[new_key] = value
            elif 'prediction_network.value_head' in key:
                new_key = key.replace('prediction_network.value_head.', '')
                value_state_dict[new_key] = value
        
        # ロード（厳密でないモードで）
        self.representation_net.load_state_dict(rep_state_dict, strict=False)
        self.policy_head.load_state_dict(policy_state_dict, strict=False)
        
        if self.use_mcts and value_state_dict:
            self.value_head.load_state_dict(value_state_dict, strict=False)
        
        print("✓ モデルのロード完了")
    
    def select_action(self, observation: np.ndarray, legal_actions: Optional[List[int]] = None) -> int:
        """
        アクションを選択
        
        Args:
            observation: 観測 [C, H, W]
            legal_actions: 合法アクションのリスト
        
        Returns:
            action: 選択されたアクション
        """
        if self.use_mcts:
            return self._select_action_mcts(observation, legal_actions)
        else:
            return self._select_action_policy(observation, legal_actions)
    
    def _select_action_policy(self, observation: np.ndarray, legal_actions: Optional[List[int]] = None) -> int:
        """ポリシーネットワークのみでアクション選択"""
        with torch.no_grad():
            # 観測をテンソルに変換
            obs_tensor = torch.from_numpy(observation).unsqueeze(0).float().to(self.device)
            
            # 表現ネットワークで潜在状態を取得
            latent_state = self.representation_net(obs_tensor)
            
            # ポリシーヘッドでアクション確率を取得
            policy_logits = self.policy_head(latent_state)
            
            # 合法アクションでマスク
            if legal_actions is not None and len(legal_actions) > 0:
                mask = torch.full_like(policy_logits, float('-inf'))
                mask[0, legal_actions] = 0
                policy_logits = policy_logits + mask
            
            # 確率分布に変換
            policy_probs = F.softmax(policy_logits, dim=-1)
            
            # 最も確率の高いアクションを選択
            action = torch.argmax(policy_probs, dim=-1).item()
            
            return action
    
    def _select_action_mcts(self, observation: np.ndarray, legal_actions: Optional[List[int]] = None) -> int:
        """MCTSでアクション選択"""
        if legal_actions is None or len(legal_actions) == 0:
            legal_actions = list(range(4))
        
        # MCTSツリーを初期化
        mcts = SimpleMCTS(
            agent=self,
            num_simulations=self.num_simulations,
            device=self.device
        )
        
        # MCTSでアクションを選択
        action = mcts.search(observation, legal_actions)
        
        return action


# =============================================================================
# MCTS (Monte Carlo Tree Search)
# =============================================================================

class MCTSNode:
    """MCTSのノード"""
    
    def __init__(self, prior: float):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}
    
    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class SimpleMCTS:
    """シンプルなMCTS実装"""
    
    def __init__(self, agent, num_simulations: int, device: str, c_puct: float = 1.0):
        self.agent = agent
        self.num_simulations = num_simulations
        self.device = device
        self.c_puct = c_puct  # exploration constant
    
    def search(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        """
        MCTSでアクションを探索
        
        Args:
            observation: 現在の観測
            legal_actions: 合法アクション
        
        Returns:
            action: 選択されたアクション
        """
        # 初期ポリシーと価値を取得
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).unsqueeze(0).float().to(self.device)
            latent_state = self.agent.representation_net(obs_tensor)
            policy_logits = self.agent.policy_head(latent_state)
            
            # 合法アクションでマスク
            mask = torch.full_like(policy_logits, float('-inf'))
            mask[0, legal_actions] = 0
            policy_logits = policy_logits + mask
            
            policy_probs = F.softmax(policy_logits, dim=-1)[0].cpu().numpy()
        
        # ルートノードを作成
        root = MCTSNode(prior=0.0)
        
        # 子ノードを展開
        for action in legal_actions:
            root.children[action] = MCTSNode(prior=policy_probs[action])
        
        # シミュレーションを実行
        for _ in range(self.num_simulations):
            self._simulate(root, observation, legal_actions)
        
        # 最も訪問回数の多いアクションを選択
        visit_counts = [(action, node.visit_count) for action, node in root.children.items()]
        action = max(visit_counts, key=lambda x: x[1])[0]
        
        return action
    
    def _simulate(self, root: MCTSNode, observation: np.ndarray, legal_actions: List[int]):
        """シミュレーションを1回実行"""
        # UCB1でアクションを選択
        action = self._select_action_ucb(root, legal_actions)
        
        node = root.children[action]
        
        # 価値を評価（簡易版：ポリシーネットワークの出力を使用）
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).unsqueeze(0).float().to(self.device)
            latent_state = self.agent.representation_net(obs_tensor)
            
            if hasattr(self.agent, 'value_head'):
                value_logits = self.agent.value_head(latent_state)
                # カテゴリカル分布から期待値を計算（簡易版）
                value = torch.softmax(value_logits, dim=-1).mean().item()
            else:
                # value headがない場合は、ポリシーの信頼度を使用
                policy_logits = self.agent.policy_head(latent_state)
                policy_probs = F.softmax(policy_logits, dim=-1)
                value = policy_probs.max().item()
        
        # バックプロパゲーション
        node.visit_count += 1
        node.value_sum += value
    
    def _select_action_ucb(self, root: MCTSNode, legal_actions: List[int]) -> int:
        """UCB1でアクションを選択"""
        best_action = legal_actions[0]
        best_ucb = float('-inf')
        
        total_visits = sum(node.visit_count for node in root.children.values())
        
        for action in legal_actions:
            node = root.children[action]
            
            # UCB1スコア
            if node.visit_count == 0:
                ucb = float('inf')
            else:
                exploitation = node.value()
                exploration = self.c_puct * node.prior * np.sqrt(total_visits) / (1 + node.visit_count)
                ucb = exploitation + exploration
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
        
        return best_action


# =============================================================================
# 評価・可視化
# =============================================================================

def render_board(board: np.ndarray, score: int, moves: int, max_tile: int):
    """盤面をテキストで表示"""
    grid_size = board.shape[0]
    
    print("\n" + "=" * (grid_size * 8 + 1))
    for row in board:
        print("|", end="")
        for cell in row:
            if cell == 0:
                print("      |", end="")
            else:
                print(f"{cell:6d}|", end="")
        print()
        print("-" * (grid_size * 8 + 1))
    
    print(f"スコア: {score} | 手数: {moves} | 最大タイル: {max_tile}")
    print("=" * (grid_size * 8 + 1))


def evaluate_agent(
    agent: GNNAgent,
    env: Game2048AnySize,
    num_episodes: int = 10,
    render: bool = False,
    save_gif: bool = False,
    gif_path: str = None
) -> dict:
    """
    エージェントを評価
    
    Args:
        agent: GNNエージェント
        env: ゲーム環境
        num_episodes: エピソード数
        render: リアルタイム描画
        save_gif: GIFとして保存
        gif_path: GIF保存パス
    
    Returns:
        stats: 統計情報
    """
    scores = []
    max_tiles = []
    moves_list = []
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"エピソード {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        obs = env.reset()
        done = False
        episode_frames = []
        
        while not done:
            # 合法アクションを取得
            legal_actions = env.get_legal_actions()
            
            if len(legal_actions) == 0:
                break
            
            # アクションを選択
            action = agent.select_action(obs, legal_actions)
            
            # ステップ実行
            obs, reward, done, info = env.step(action)
            
            # 描画
            if render:
                render_board(env.board, info['score'], info['moves'], info['max_tile'])
                time.sleep(0.1)
            
            # GIF用のフレームを保存
            if save_gif:
                episode_frames.append(env.board.copy())
        
        # エピソード終了
        scores.append(info['score'])
        max_tiles.append(info['max_tile'])
        moves_list.append(info['moves'])
        
        print(f"\n最終結果:")
        print(f"  スコア: {info['score']}")
        print(f"  最大タイル: {info['max_tile']}")
        print(f"  手数: {info['moves']}")
        
        # GIFを保存
        if save_gif and gif_path and len(episode_frames) > 0:
            save_game_as_gif(episode_frames, scores[-1], gif_path)
    
    # 統計を計算
    stats = {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'mean_max_tile': np.mean(max_tiles),
        'max_tile_counts': {tile: max_tiles.count(tile) for tile in set(max_tiles)},
        'mean_moves': np.mean(moves_list),
    }
    
    return stats


def save_game_as_gif(frames: List[np.ndarray], score: int, output_path: str):
    """ゲームをGIFとして保存"""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import colors
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # カラーマップ
    cmap = colors.ListedColormap(['#CDC1B4', '#EEE4DA', '#EDE0C8', '#F2B179',
                                   '#F59563', '#F67C5F', '#F65E3B', '#EDCF72',
                                   '#EDCC61', '#EDC850', '#EDC53F', '#EDC22E'])
    
    def update(frame_idx):
        ax.clear()
        board = frames[frame_idx]
        grid_size = board.shape[0]
        
        # ボードを描画
        log_board = np.where(board > 0, np.log2(board).astype(int), 0)
        im = ax.imshow(log_board, cmap=cmap, vmin=0, vmax=11)
        
        # タイルの値を表示
        for i in range(grid_size):
            for j in range(grid_size):
                if board[i, j] > 0:
                    ax.text(j, i, str(int(board[i, j])), ha='center', va='center',
                           fontsize=20 - grid_size, fontweight='bold')
        
        ax.set_title(f'2048 GNN - Move {frame_idx + 1}/{len(frames)}\nScore: {score}',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
    
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=200)
    anim.save(output_path, writer='pillow', fps=5)
    plt.close()
    
    print(f"✓ GIFを保存しました: {output_path}")


def print_statistics(stats: dict, grid_size: int):
    """統計情報を表示"""
    print("\n" + "="*60)
    print(f"統計情報 (盤面サイズ: {grid_size}×{grid_size})")
    print("="*60)
    print(f"平均スコア:     {stats['mean_score']:.2f} ± {stats['std_score']:.2f}")
    print(f"最高スコア:     {stats['max_score']:.0f}")
    print(f"最低スコア:     {stats['min_score']:.0f}")
    print(f"平均最大タイル: {stats['mean_max_tile']:.0f}")
    print(f"平均手数:       {stats['mean_moves']:.1f}")
    print(f"\n最大タイル達成回数:")
    for tile, count in sorted(stats['max_tile_counts'].items(), reverse=True):
        print(f"  {tile:4d}: {count}回")
    print("="*60)


# =============================================================================
# メイン
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GNN 2048 汎用サイズエミュレータ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 3×3盤面で10エピソード実行
  python gnn_any_size_emulator.py --grid-size 3 --episodes 10
  
  # 5×5盤面でリアルタイム描画付き
  python gnn_any_size_emulator.py --grid-size 5 --episodes 5 --render
  
  # 6×6盤面でGIF保存
  python gnn_any_size_emulator.py --grid-size 6 --episodes 3 --save-gif
  
  # MCTSを使用（50シミュレーション）
  python gnn_any_size_emulator.py --grid-size 4 --use-mcts --num-simulations 50
  
  # MCTSで強力な推論（200シミュレーション）
  python gnn_any_size_emulator.py --grid-size 4 --use-mcts --num-simulations 200 --episodes 3
  
  # カスタムモデルを使用
  python gnn_any_size_emulator.py --grid-size 4 --model-path path/to/model.pth.tar
        """
    )
    
    parser.add_argument('--grid-size', type=int, default=4,
                       help='盤面のサイズ (3〜8を推奨, デフォルト: 4)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='実行するエピソード数 (デフォルト: 10)')
    parser.add_argument('--model-path', type=str,
                       default='/opendilab/2048GNN/LightZero/zoo/game_2048/config/data_gnn_stochastic_mz/gnn_simple_success1/ckpt/iteration_79400.pth.tar',
                       help='学習済みモデルのパス（省略時は grid-size に応じて自動選択）')
    parser.add_argument('--render', action='store_true',
                       help='リアルタイムで盤面を表示')
    parser.add_argument('--save-gif', action='store_true',
                       help='最初のエピソードをGIFとして保存')
    parser.add_argument('--gif-path', type=str,
                       default='./gnn_2048_custom_size.gif',
                       help='GIFの保存パス')
    parser.add_argument('--use-mcts', action='store_true',
                       help='MCTSを使用して推論を強化（より良い性能、より遅い）')
    parser.add_argument('--num-simulations', type=int, default=50,
                       help='MCTSのシミュレーション回数 (デフォルト: 50, 推奨: 50-200)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='使用するデバイス')
    
    args = parser.parse_args()
    
    # モデルパスの自動選択
    if args.model_path is None:
        if args.grid_size == 3:
            args.model_path = '/opendilab/2048GNN/LightZero/zoo/game_2048/config/data_gnn_stochastic_mz_3x3/game_2048_gnn_3x3_npct-2_ns50_upc100_rer0.0_bs256_gnn2L96D_adjacent_seed0_resume_251008_215736/ckpt/iteration_60000.pth.tar'
        elif args.grid_size == 4:
            # 4×4のデフォルトモデルを探す（最新の最適化版を優先）
            possible_paths = [
                '/opendilab/2048GNN/LightZero/data_gnn_stochastic_mz_optimized/game_2048_gnn_opt_npct-2_ns100_upc200_rer0.0_bs512_gnn3L128D_sparse_seed0_251011_032638/ckpt/ckpt_best.pth.tar',
                '/opendilab/2048GNN/LightZero/data_gnn_stochastic_mz_optimized/game_2048_gnn_opt_npct-2_ns100_upc200_rer0.0_bs512_gnn3L128D_sparse_seed0_251011_032515/ckpt/ckpt_best.pth.tar',
                '/opendilab/2048GNN/LightZero/data_gnn_stochastic_mz/game_2048_gnn_npct-2_ns100_upc200_rer0.0_bs512_gnn3L128D_sparse_seed0_251007_230441/ckpt/ckpt_best.pth.tar',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    args.model_path = path
                    break
            if args.model_path is None:
                print(f"エラー: 4×4用の学習済みモデルが見つかりません。")
                print("--model-path オプションでモデルパスを指定してください。")
                sys.exit(1)
        else:
            print(f"警告: grid-size {args.grid_size} 用のデフォルトモデルはありません。")
            print("4×4モデルで試行します（転移学習）...")
            # 4×4モデルをフォールバックとして使用
            possible_paths = [
                '/opendilab/2048GNN/LightZero/data_gnn_stochastic_mz_optimized/game_2048_gnn_opt_npct-2_ns100_upc200_rer0.0_bs512_gnn3L128D_sparse_seed0_251011_032638/ckpt/ckpt_best.pth.tar',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    args.model_path = path
                    break
            if args.model_path is None:
                print(f"エラー: フォールバック用のモデルも見つかりません。")
                print("--model-path オプションでモデルパスを指定してください。")
                sys.exit(1)
    
    # モデルパスの存在確認
    if not os.path.exists(args.model_path):
        print(f"エラー: モデルファイルが見つかりません: {args.model_path}")
        sys.exit(1)
    
    # ヘッダー
    print("="*60)
    print("GNN 2048 汎用サイズエミュレータ")
    print("="*60)
    print(f"盤面サイズ:       {args.grid_size}×{args.grid_size}")
    print(f"エピソード数:     {args.episodes}")
    print(f"モデルパス:       {os.path.basename(args.model_path)}")
    print(f"推論モード:       {'MCTS' if args.use_mcts else 'ポリシーのみ'}")
    if args.use_mcts:
        print(f"シミュレーション回数: {args.num_simulations}")
    print(f"デバイス:         {args.device}")
    print(f"リアルタイム描画: {'有効' if args.render else '無効'}")
    print(f"GIF保存:          {'有効' if args.save_gif else '無効'}")
    print("="*60)
    
    # デバイスの確認
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDAが利用できません。CPUを使用します。")
        args.device = 'cpu'
    
    # 環境とエージェントを初期化
    env = Game2048AnySize(grid_size=args.grid_size)
    agent = GNNAgent(
        model_path=args.model_path,
        grid_size=args.grid_size,
        device=args.device,
        use_mcts=args.use_mcts,
        num_simulations=args.num_simulations
    )
    
    # 評価実行
    print("\n評価を開始します...\n")
    stats = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=args.episodes,
        render=args.render,
        save_gif=args.save_gif,
        gif_path=args.gif_path if args.save_gif else None
    )
    
    # 統計情報を表示
    print_statistics(stats, args.grid_size)
    
    print("\n✓ 評価完了！")


if __name__ == '__main__':
    main()
