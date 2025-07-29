from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.torch_utils import MLP
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import MZNetworkOutput
# 元の実装にあるヘルパー関数をインポート（必要に応じて）
from .utils import renormalize, get_params_mean, get_dynamic_mean, get_reward_mean


class GNNLayer(nn.Module):
    """
    Overview:
        シンプルなGNNレイヤー。各ノードは隣接ノードからメッセージを受け取り、自身の特徴を更新します。
        バッチ処理に対応するため、隣接行列を用いた行列演算で実装されています。
    """

    def __init__(self, in_features: int, out_features: int):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            - node_features (:obj:`torch.Tensor`): ノード特徴テンソル (B, N, in_features)
            - adj_matrix (:obj:`torch.Tensor`): 隣接行列 (B, N, N)
        Returns:
            - (:obj:`torch.Tensor`): 更新されたノード特徴テンソル (B, N, out_features)
        """
        # 線形変換でメッセージを準備
        support = self.linear(node_features)
        # 隣接行列を使ってメッセージを集約
        output = torch.bmm(adj_matrix, support)
        return output


class GNNRepresentationNetwork(nn.Module):
    """
    Overview:
        GNNベースの表現ネットワーク。盤面の観測（画像形式）をグラフ構造に変換し、
        GNNで処理して潜在表現ベクトル（latent_state）を生成します。
    """

    def __init__(
            self,
            observation_shape: SequenceType,
            node_embed_dim: int = 64,
            gnn_hidden_dim: int = 128,
            gnn_num_layers: int = 3,
            readout_output_dim: int = 256,
            activation: nn.Module = nn.ReLU(inplace=True)
    ):
        super(GNNRepresentationNetwork, self).__init__()
        self.observation_shape = observation_shape
        # タイルの値（logスケール）をベクトルに埋め込むためのMLP
        # チャンネル数がタイルの情報を表す
        self.node_embedding = MLP(
            in_channels=observation_shape[0],
            hidden_channels=node_embed_dim,
            out_channels=node_embed_dim,
            layer_num=2,
            activation=activation,
        )

        # GNNレイヤー
        gnn_layers = [GNNLayer(node_embed_dim, gnn_hidden_dim)]
        gnn_layers += [GNNLayer(gnn_hidden_dim, gnn_hidden_dim) for _ in range(gnn_num_layers - 1)]
        self.gnn_layers = nn.ModuleList(gnn_layers)

        # Readout層 (全ノードの特徴を集約してグラフ全体の表現を生成)
        self.readout = MLP(
            in_channels=gnn_hidden_dim,
            hidden_channels=gnn_hidden_dim * 2,
            out_channels=readout_output_dim,
            layer_num=2,
            activation=activation,
        )
        self.activation = activation

    def _build_graph(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """観測から動的にグラフ（ノード特徴と隣接行列）を構築"""
        
        if obs.dim() == 5:
            B_orig, S, C_in, H_in, W_in = obs.shape
            obs = obs.view(B_orig * S, C_in, H_in, W_in)
        
        
        B, C, H, W = obs.shape
        num_nodes = H * W

        # (B, C, H, W) -> (B, H*W, C)
        node_features = obs.permute(0, 2, 3, 1).reshape(B, num_nodes, C)
        # ノード特徴を埋め込み
        embedded_nodes = self.node_embedding(node_features)  # (B, N, embed_dim)

        # 隣接行列を動的に構築
        adj = torch.zeros(B, num_nodes, num_nodes, device=obs.device)
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                if i > 0:
                    adj[:, idx, idx - W] = 1
                if i < H - 1:
                    adj[:, idx, idx + W] = 1
                if j > 0:
                    adj[:, idx, idx - 1] = 1
                if j < W - 1:
                    adj[:, idx, idx + 1] = 1
        
        # 正規化 (D^-1 * A)
        adj_sum = torch.sum(adj, dim=2, keepdim=True)
        adj_norm = adj / (adj_sum + 1e-6) # ゼロ除算を防止

        return embedded_nodes, adj_norm

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        node_features, adj_matrix = self._build_graph(obs)

        for layer in self.gnn_layers:
            node_features = self.activation(layer(node_features, adj_matrix))

        # Global Mean PoolingによるReadout
        graph_representation = node_features.mean(dim=1)
        latent_state = self.readout(graph_representation)
        return latent_state

@MODEL_REGISTRY.register('StochasticMuZeroModelGNN')
class StochasticMuZeroModelGNN(nn.Module):
    """
    Overview:
        GNN版Stochastic MuZeroモデル。
        CNNベースのコンポーネントをGNNベースに置き換えることで、
        盤面サイズに依存しない汎用的なモデルを実現します。
    """

    def __init__(
            self,
            observation_shape: SequenceType = (1, 4, 4), # (C, H, W)
            action_space_size: int = 4, # 2048ゲームのアクション数
            chance_space_size: int = 2,
            # GNN関連のハイパーパラメータ
            node_embed_dim: int = 64,
            gnn_hidden_dim: int = 128,
            gnn_num_layers: int = 3,
            latent_state_dim: int = 256,
            # MLP Head関連のハイパーパラメータ
            mlp_hidden_dim: int = 256,
            reward_support_size: int = 601,
            value_support_size: int = 601,
            self_supervised_learning_loss: bool = False,
            categorical_distribution: bool = True,
            activation: nn.Module = nn.ReLU(inplace=True),
            state_norm: bool = False,
            *args,
            **kwargs
    ):
        super(StochasticMuZeroModelGNN, self).__init__()
        self.action_space_size = action_space_size
        self.chance_space_size = chance_space_size
        self.categorical_distribution = categorical_distribution
        if self.categorical_distribution:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size
        else:
            self.reward_support_size = 1
            self.value_support_size = 1
        
        self.state_norm = state_norm
        self.self_supervised_learning_loss = self_supervised_learning_loss

        # 1. 表現ネットワーク (GNN)
        self.representation_network = GNNRepresentationNetwork(
            observation_shape=observation_shape,
            node_embed_dim=node_embed_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            readout_output_dim=latent_state_dim,
            activation=activation
        )

        # 2. 予測ネットワーク (MLP)
        self.prediction_network = MLP(
            in_channels=latent_state_dim,
            hidden_channels=mlp_hidden_dim,
            out_channels=action_space_size + self.value_support_size,
            layer_num=3,
            activation=activation
        )
        
        # 3. ダイナミクスネットワーク (MLP)
        self.dynamics_network = MLP(
            # 入力は (latent_state + one-hot encoded chance)
            in_channels=latent_state_dim + chance_space_size,
            hidden_channels=mlp_hidden_dim,
            # 出力は (next_latent_state + reward)
            out_channels=latent_state_dim + self.reward_support_size,
            layer_num=3,
            activation=activation
        )
        
        # 4. Afterstate ダイナミクスネットワーク (MLP)
        self.afterstate_dynamics_network = MLP(
            # 入力は (latent_state + one-hot encoded action)
            in_channels=latent_state_dim + action_space_size,
            hidden_channels=mlp_hidden_dim,
            # 出力は (next_afterstate + reward)
            out_channels=latent_state_dim + self.reward_support_size,
            layer_num=3,
            activation=activation
        )

        # 5. Afterstate 予測ネットワーク (MLP)
        self.afterstate_prediction_network = MLP(
            in_channels=latent_state_dim,
            hidden_channels=mlp_hidden_dim,
            out_channels=chance_space_size + self.value_support_size,
            layer_num=3,
            activation=activation
        )
        # 6. Chance Encoderネットワーク (自己教師あり学習用)
        # 観測からランダム事象を予測するための小さなネットワーク
        # (省略) ChanceEncoderや自己教師あり学習の部分も必要に応じてGNNベースに適合させます。
        self.chance_encoder = MLP(
                in_channels=latent_state_dim,
                hidden_channels=mlp_hidden_dim,
                out_channels=self.chance_space_size,
                layer_num=2,
                activation=activation
            )
        # この例では主要なネットワークの置き換えに焦点を当てています。
        # 7. Projection Head (自己教師あり学習用)
        # 潜在表現を別の特徴空間に射影するためのネットワーク
        self.projection_head = MLP(
            in_channels=latent_state_dim,
            hidden_channels=mlp_hidden_dim,
            out_channels=latent_state_dim,  # 同じ次元に射影するのが一般的
            layer_num=2,
            activation=activation
        )
        # 8. Prediction Head (自己教師あり学習用)
        # projectメソッドとペアで使われる予測ヘッド
        self.prediction_head_ssl = MLP(
            in_channels=latent_state_dim,
            hidden_channels=mlp_hidden_dim,
            out_channels=latent_state_dim,
            layer_num=2,
            activation=activation
)
    def get_dynamic_mean(self) -> float:
        """
        ダイナミクスネットワークの出力の平均値を取得する（ログ用）
        """
        return get_dynamic_mean(self)

    def get_reward_mean(self) -> float:
        """
        報酬予測の出力の平均値を取得する（ログ用）
        """
        return get_reward_mean(self)
    # StochasticMuZeroModelGNN クラス内に追加
    def prediction_ssl(self, project_output: torch.Tensor) -> torch.Tensor:
        """
        自己教師あり学習のため、射影された特徴量からターゲットを予測する
        """
        return self.prediction_head_ssl(project_output)
    def project(self, latent_state: torch.Tensor, with_grad: bool = True) -> torch.Tensor:
        """
        自己教師あり学習のため、潜在状態を別の空間に射影する

        Arguments:
            - latent_state (:obj:`torch.Tensor`): 潜在状態テンソル
            - with_grad (:obj:`bool`): 勾配計算を有効にするか
        Returns:
            - (:obj:`torch.Tensor`): 射影されたテンソル
        """
        if with_grad:
            return self.projection_head(latent_state)
        else:
            # ターゲットを計算する場合など、勾配を計算しない
            with torch.no_grad():
                return self.projection_head(latent_state)
    def chance_encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        観測データから、ランダム事象（chance）の発生を予測するエンコーダー

        Arguments:
            - obs (:obj:`torch.Tensor`): 観測データ（盤面）のテンソル
        Returns:
            - Tuple[torch.Tensor, torch.Tensor]: chanceの予測ロジットと、そのone-hot表現
        """
        # 観測を潜在表現に変換
        latent_state = self._representation(obs)

        # 潜在表現からchanceのロジットを予測
        chance_logits = self.chance_encoder(latent_state)

        # 最も可能性の高いchanceをone-hotベクトルに変換
        chance_one_hot = F.one_hot(torch.argmax(chance_logits, dim=1), self.chance_space_size).float()

        return chance_logits, chance_one_hot
    def _support_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        """
        カテゴリカル分布のlogitsをスカラーの期待値に変換します。
        """
        # support_sizeが1の場合は、分布ではなく直接値が与えられているとみなします。
        if self.reward_support_size == 1:
            return logits.squeeze(-1)

        # 報酬のサポート範囲を定義します。これはMuZeroのハイパーパラメータです。
        # ここでは一般的な例として、-300から300までの整数値をサポートとして使用します。
        support = torch.arange(
            -(self.reward_support_size // 2),
            self.reward_support_size // 2 + 1,
            device=logits.device
        ).float()

        # softmaxで確率に変換し、サポートとの加重平均（期待値）を計算します。
        probabilities = torch.softmax(logits, dim=1)
        scalar = torch.sum(probabilities * support, dim=1)
        return scalar

    def _representation(self, observation: torch.Tensor) -> torch.Tensor:
        latent_state = self.representation_network(observation)
        if self.state_norm:
            latent_state = renormalize(latent_state)
        return latent_state

    def _prediction(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.prediction_network(latent_state)
        policy_logits = output[:, :self.action_space_size]
        value = output[:, self.action_space_size:]
        return policy_logits, value

    def _afterstate_prediction(self, afterstate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.afterstate_prediction_network(afterstate)
        # afterstateからはchanceの予測を行う
        policy_logits = output[:, :self.chance_space_size]
        value = output[:, self.chance_space_size:]
        return policy_logits, value

    def _dynamics(self, latent_state: torch.Tensor, chance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # chanceをone-hotベクトルに変換
        chance_onehot = F.one_hot(chance.long(), num_classes=self.chance_space_size).float()
        state_chance_encoding = torch.cat((latent_state, chance_onehot), dim=1)
        
        output = self.dynamics_network(state_chance_encoding)
        next_latent_state = output[:, :latent_state.shape[1]]
        reward = output[:, latent_state.shape[1]:]

        if self.state_norm:
            next_latent_state = renormalize(next_latent_state)
        return next_latent_state, reward

    def _afterstate_dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if action.dim() == 2:
            action = action.squeeze(1)
        
        # actionをone-hotベクトルに変換
        action_onehot = F.one_hot(action.long(), num_classes=self.action_space_size).float()
        state_action_encoding = torch.cat((latent_state, action_onehot), dim=1)
        
        output = self.afterstate_dynamics_network(state_action_encoding)
        next_afterstate = output[:, :latent_state.shape[1]]
        reward = output[:, latent_state.shape[1]:]

        if self.state_norm:
            next_afterstate = renormalize(next_afterstate)
        return next_afterstate, reward

    def initial_inference(self, obs: torch.Tensor) -> MZNetworkOutput:
        batch_size = obs.size(0)
        latent_state = self._representation(obs)
        policy_logits, value = self._prediction(latent_state)

        # 初期ステップの報酬（分布）はゼロ
        reward_logits = torch.zeros(batch_size, self.reward_support_size, device=obs.device)

        if self.training:
            # 学習時は分布をそのまま返す
            return MZNetworkOutput(value, reward_logits, policy_logits, latent_state)
        else:
            # 評価時はスカラーに変換して list で返す
            reward_scalar = self._support_to_scalar(reward_logits)
            return MZNetworkOutput(value, reward_scalar.tolist(), policy_logits, latent_state)

    def recurrent_inference(self, state: torch.Tensor, option: torch.Tensor, afterstate: bool = False) -> MZNetworkOutput:
        if afterstate:
            # stateはafterstate, optionはchance
            next_latent_state, reward_logits = self._dynamics(state, option)
            policy_logits, value = self._prediction(next_latent_state)
        else:
            # stateはlatent_state, optionはaction
            next_latent_state, reward_logits = self._afterstate_dynamics(state, option)
            policy_logits, value = self._afterstate_prediction(next_latent_state)

        if self.training:
            # 学習時は分布をそのまま返す
            return MZNetworkOutput(value, reward_logits, policy_logits, next_latent_state)
        else:
            # 評価時はスカラーに変換して list で返す
            reward_scalar = self._support_to_scalar(reward_logits)
            return MZNetworkOutput(value, reward_scalar.tolist(), policy_logits, next_latent_state)

    def get_params_mean(self) -> float:
        return get_params_mean(self)