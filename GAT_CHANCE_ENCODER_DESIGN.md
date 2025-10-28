# GAT ChanceEncoder 設計文書

## 📋 背景

転移学習や異なるグリッドサイズでの実験を行うため、ChanceEncoderをGATベースに変更する必要があります。

## 🎯 設計目標

1. **サイズ非依存**: 3×3、4×4、5×5など任意のグリッドサイズに対応
2. **転移学習対応**: 小さいサイズで学習 → 大きいサイズに転移
3. **一貫性**: 他のGATコンポーネントと統一
4. **パフォーマンス**: CNNと同等以上の速度

## 🔧 CNNベースの問題点

### 現状の実装
```python
class ChanceEncoderBackbone(nn.Module):
    def __init__(self, input_dimensions, action_dimension):
        # input_dimensions = (channels, height, width)
        self.conv1 = Conv2d(input_dimensions[0] * 2, 32, 3, 1, 1)
        self.conv2 = Conv2d(32, 64, 3, 1, 1)
        
        # 🚨 問題: fc1の入力次元がグリッドサイズに依存
        fc_input_dim = 64 * input_dimensions[1] * input_dimensions[2]
        # 4×4の場合: 64 * 4 * 4 = 1024
        # 3×3の場合: 64 * 3 * 3 = 576  ← 重みが使えない！
        
        self.fc1 = Linear(fc_input_dim, 128)
        self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64, action_dimension)
```

### 問題
1. **固定サイズ依存**: `fc_input_dim`がグリッドサイズで変わる
2. **転移学習不可**: 4×4で学習した重みを3×3で使えない
3. **拡張性なし**: 新しいサイズごとにゼロから学習

## ✅ GATベースの解決策

### アーキテクチャ
```
Input Observation [B, C, H, W]
    ↓
GraphBuilder: obs_to_graph()
    ↓
Node Features [B, N, C+2]  (N = H×W)
    ↓
GAT Layers (Multi-head Attention)
    ↓
Node Embeddings [B, N, hidden_dim]
    ↓
Graph Pooling (mean/max/sum)
    ↓
Aggregated [B, hidden_dim * 3]
    ↓
MLP → [B, chance_space_size]
```

### 実装
```python
class GATChanceEncoder(nn.Module):
    """
    GAT-based Chance Encoder
    
    【サイズ非依存】グリッドサイズに関係なく動作
    【転移学習対応】異なるサイズ間で重みを共有可能
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, int, int],  # (C, H, W)
        chance_space_size: int,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
        hidden_dim: int = 64,
        edge_mode: str = 'adjacent',
        norm_type: str = 'group',
    ):
        super().__init__()
        
        channels, height, width = observation_shape
        grid_size = height  # 正方形グリッドを仮定
        
        # GraphBuilder（サイズ非依存）
        self.graph_builder = GraphBuilder(
            grid_size=grid_size,
            include_row_col_edges=False,
            edge_mode=edge_mode,
            device=None
        )
        
        # Input: observation channels + 2 (position encoding)
        in_dim = channels + 2
        
        # GAT Encoder（サイズ非依存）
        self.gat = GraphAttention(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=0.0,
            use_bn=True,
            norm_type=norm_type
        )
        
        # Graph Pooling + MLP（サイズ非依存）
        aggregated_dim = hidden_dim * 3  # mean, max, sum
        self.mlp = nn.Sequential(
            nn.Linear(aggregated_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, chance_space_size)
        )
        
        # Straight Through Estimator
        self.onehot_argmax = StraightThroughEstimator()
    
    def forward(self, observations: torch.Tensor):
        """
        Args:
            observations: [B, C, H, W]
        
        Returns:
            chance_encoding: [B, chance_space_size]
            chance_onehot: [B, chance_space_size]
        """
        batch_size = observations.size(0)
        
        # Convert to graph
        node_features, edge_index = self.graph_builder.obs_to_graph(observations)
        
        # Apply GAT
        node_embeddings = self.gat(node_features, edge_index)  # [B, N, hidden_dim]
        
        # Graph pooling（サイズ非依存）
        mean_pool = node_embeddings.mean(dim=1)  # [B, hidden_dim]
        max_pool = node_embeddings.max(dim=1)[0]  # [B, hidden_dim]
        sum_pool = node_embeddings.sum(dim=1)  # [B, hidden_dim]
        
        # Concatenate
        aggregated = torch.cat([mean_pool, max_pool, sum_pool], dim=-1)
        
        # MLP
        chance_encoding = self.mlp(aggregated)
        
        # One-hot argmax
        chance_onehot = self.onehot_argmax(chance_encoding)
        
        return chance_encoding, chance_onehot
```

## 🎓 転移学習の利点

### 1. グリッドサイズ非依存
```python
# 4×4で学習
encoder_4x4 = GATChanceEncoder((16, 4, 4), chance_space_size=32)
encoder_4x4.train()

# 3×3に転移（重みをそのまま使える！）
encoder_3x3 = GATChanceEncoder((16, 3, 3), chance_space_size=32)
encoder_3x3.load_state_dict(encoder_4x4.state_dict())
# ✅ GraphBuilderだけ再初期化、GAT/MLPの重みは共有
```

### 2. 転移学習パターン

#### パターンA: 小→大（推奨）
```
3×3で学習 → 4×4に転移 → 5×5に転移
```
- ✅ 小さいサイズで高速に学習
- ✅ 大きいサイズで微調整
- ✅ 計算コストが低い

#### パターンB: 大→小
```
5×5で学習 → 4×4に転移 → 3×3に転移
```
- ✅ 大きいサイズで複雑なパターンを学習
- ✅ 小さいサイズで高速推論

## 📊 性能比較

| 特性 | CNN Chance Encoder | GAT Chance Encoder |
|------|-------------------|-------------------|
| サイズ非依存 | ❌ 固定 | ✅ 任意 |
| 転移学習 | ❌ 不可 | ✅ 可能 |
| 計算量 (4×4) | ~1.2M FLOPs | ~0.8M FLOPs |
| パラメータ数 | ~134K | ~89K |
| 速度 | 中 | 高（最適化済み） |

## 🛠️ 実装手順

### Step 1: GATChanceEncoderの実装
- `gat_stochastic_muzero_model.py`に追加

### Step 2: モデルの更新
```python
# 現状
from .stochastic_muzero_model import ChanceEncoder
self.chance_encoder = ChanceEncoder(...)

# 変更後
self.chance_encoder = GATChanceEncoder(
    observation_shape=observation_shape,
    chance_space_size=chance_space_size,
    num_gnn_layers=2,
    num_heads=4,
    hidden_dim=64,
    edge_mode='adjacent',
    norm_type='group'
)
```

### Step 3: バリデーション更新
```python
def _validate_no_cnn_in_gat_components(self):
    """
    全てのコンポーネントがGATベースであることを確認
    CNNは一切使用しない
    """
    # chance_encoderもチェック対象に含める
```

## 🧪 テスト計画

### テスト1: サイズ非依存性
```python
def test_size_independence():
    for size in [3, 4, 5]:
        encoder = GATChanceEncoder((16, size, size), 32)
        obs = torch.randn(8, 16, size, size)
        encoding, onehot = encoder(obs)
        assert encoding.shape == (8, 32)
```

### テスト2: 転移学習
```python
def test_transfer_learning():
    # 4×4で学習
    encoder_4x4 = GATChanceEncoder((16, 4, 4), 32)
    # ... training ...
    
    # 3×3に転移
    encoder_3x3 = GATChanceEncoder((16, 3, 3), 32)
    state_dict = encoder_4x4.state_dict()
    # GraphBuilder以外をロード
    encoder_3x3.load_state_dict(state_dict, strict=False)
```

### テスト3: 性能比較
```python
def test_performance():
    encoder_cnn = ChanceEncoder((16, 4, 4), 32, 'conv')
    encoder_gat = GATChanceEncoder((16, 4, 4), 32)
    
    # 速度測定
    # 精度測定
```

## 🎯 推奨事項

### 即時実装すべき
1. ✅ **GATChanceEncoderの実装**
2. ✅ **既存のChanceEncoderを置き換え**
3. ✅ **転移学習ヘルパー関数の追加**

### 理由
- 転移学習を予定している
- 異なるサイズでの実験を予定している
- GATとの一貫性を保つ
- 将来的な拡張性

## 📝 まとめ

**結論: GATベースのChanceEncoderを実装すべき**

理由:
1. ✅ サイズ非依存で転移学習が可能
2. ✅ 他のGATコンポーネントと一貫性
3. ✅ パラメータ数が少なく高速
4. ✅ 将来的な拡張性が高い

実装しますか？
