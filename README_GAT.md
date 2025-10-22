# GAT-based Stochastic MuZero for 2048

Graph Attention Network (GAT)を使用した2048ゲーム用のStochastic MuZero実装です。

## 概要

このプロジェクトは、従来のCNNベースのMuZeroをGraph Neural Network (GAT)ベースに置き換えたものです。GATは**マルチヘッドアテンション機構**を使用してグラフノード間の関係を学習します。

### 主な特徴

- ✅ **完全にGATベース**: representation, dynamics, predictionネットワークはすべてGATを使用
- ✅ **マルチヘッドアテンション**: 複数のアテンションヘッドで異なる関係性を学習
- ✅ **エッジモード最適化**: 'adjacent', 'sparse', 'full'から選択可能
- ✅ **CNN不使用**: chance_encoder以外でCNNを完全に排除（バリデーション付き）
- ✅ **既存インフラと互換**: MuZeroのトレーニングパイプラインをそのまま使用可能

## アーキテクチャ

### GATコンポーネント

1. **GraphAttentionConv**: アテンションベースのメッセージパッシング層
   - マルチヘッドアテンション機構
   - ソースとターゲットノードの特徴を結合
   - LeakyReLU活性化関数とsoftmax正規化

2. **GraphAttention**: 複数のGATConv層をスタック
   - レイヤー正規化（LayerNorm）
   - ドロップアウト
   - 残差接続

3. **GATRepresentationNetwork**: 観測を潜在状態に変換
4. **GATDynamicsNetwork**: 状態遷移を予測
5. **GATPredictionNetwork**: 価値とポリシーを予測

### ハイパーパラメータ

```python
# GAT固有のパラメータ
num_gnn_layers = 3        # GAT層の数
gnn_hidden_dim = 128      # 隠れ層の次元数（ヘッドごと）
num_heads = 4             # アテンションヘッドの数
edge_mode = 'sparse'      # エッジ接続モード
gnn_dropout = 0.0         # ドロップアウト率
```

### エッジモード

グラフの接続性を制御：

- **adjacent** (~56エッジ): 最速、4近傍のみ
- **sparse** (~88エッジ): バランス型、4近傍+距離2
- **full** (~200エッジ): 最遅、同じ行/列のすべてのペア

推奨: `sparse` (速度と精度のバランスが良い)

## ファイル構成

```
LightZero/
├── lzero/
│   └── model/
│       ├── gat_utils.py                      # GAT実装
│       └── gat_stochastic_muzero_model.py    # GATベースMuZeroモデル
└── zoo/
    └── game_2048/
        └── config/
            └── stochastic_muzero_2048_gat_config.py  # GAT設定

2048GNN/
├── test_gat_model_simple.py          # モデルテスト
├── quick_gat_training_test.py        # トレーニングテスト
└── README_GAT.md                      # このファイル
```

## 使用方法

### 1. モデルのテスト

```bash
cd /opendilab/2048GNN
python test_gat_model_simple.py
```

期待される出力:
```
✅ All tests passed! GAT model is working correctly.

Model Summary:
  - Total parameters: 2,541,768
  - Trainable parameters: 2,541,768
  - Attention heads: 4
  - GAT layers: 3
  - Hidden dim: 128
  - Edge mode: sparse
```

### 2. クイックトレーニングテスト

```bash
cd /opendilab/2048GNN
python quick_gat_training_test.py
```

### 3. 本格的なトレーニング

```bash
cd /opendilab/2048GNN/LightZero
python zoo/game_2048/config/stochastic_muzero_2048_gat_config.py
```

### 4. Pythonから使用

```python
from lzero.model.gat_stochastic_muzero_model import GATStochasticMuZeroModel

model = GATStochasticMuZeroModel(
    observation_shape=(16, 4, 4),
    action_space_size=4,
    chance_space_size=32,
    num_channels=128,
    num_gnn_layers=3,
    num_heads=4,  # GAT: アテンションヘッド数
    grid_size=4,
    edge_mode='sparse',
)

# 推論
output = model.initial_inference(obs)
```

## GNN (GraphSAGE) vs GAT 比較

| 特徴 | GraphSAGE | GAT |
|------|-----------|-----|
| 集約方法 | mean/max/sum | アテンション重み付け |
| ノード重要度 | 均等または固定 | 学習可能 |
| 計算量 | 低い | 中程度（マルチヘッド） |
| 表現力 | 中 | 高 |
| パラメータ数 | 少ない | 多い（アテンション機構分） |

### パラメータ数比較

- **GNN (GraphSAGE)**: ~2.4M パラメータ
- **GAT (4 heads)**: ~2.5M パラメータ
- **GAT (8 heads)**: ~2.7M パラメータ

## 技術的詳細

### アテンション機構

GATは各エッジに対してアテンション係数を計算します：

1. **線形変換**: 入力特徴を変換
2. **アテンションスコア**: ソース・ターゲット特徴を結合して計算
3. **LeakyReLU**: 非線形活性化
4. **Softmax**: 目的ノードごとに正規化
5. **重み付け集約**: アテンション係数で重み付けして集約

```python
# 疑似コード
alpha = softmax(LeakyReLU(a^T [W*h_i || W*h_j]))
h_i' = Σ_j alpha_ij * W * h_j
```

### バリデーション

モデル初期化時に自動的にCNN使用をチェック：

```python
def _validate_no_cnn_in_gat_components(self):
    """GAT部分でCNNが使用されていないことを確認"""
    # chance_encoder以外でConv2d, ResBlock, BatchNorm2dを検出すると例外
```

## トラブルシューティング

### エラー: "必須GATコンポーネントが見つかりません"

→ モデルの登録を確認:
```bash
grep -r "GATStochasticMuZeroModel" LightZero/lzero/model/__init__.py
```

### エラー: "GAT部分でCNNレイヤーが検出されました"

→ representation/dynamics/predictionネットワークでCNNが使用されています。
   chance_encoderのみCNN使用が許可されています。

### メモリ不足

- `batch_size`を減らす (512 → 256)
- `num_heads`を減らす (4 → 2)
- `edge_mode`を'adjacent'に変更

## パフォーマンスチューニング

### 速度優先

```python
edge_mode = 'adjacent'
num_heads = 2
num_gnn_layers = 2
```

### 精度優先

```python
edge_mode = 'full'
num_heads = 8
num_gnn_layers = 4
```

### バランス型（推奨）

```python
edge_mode = 'sparse'
num_heads = 4
num_gnn_layers = 3
```

## 参考文献

1. **Graph Attention Networks** (Veličković et al., 2018)
   - https://arxiv.org/abs/1710.10903
   
2. **Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model** (Schrittwieser et al., 2020)
   - MuZero原論文

3. **Stochastic MuZero** (Antonoglou et al., 2022)
   - 確率的環境への拡張

## ライセンス

このプロジェクトは元のLightZeroプロジェクトと同じライセンスに従います。

## 作成日

2024年10月22日

---

**Note**: このREADMEは、GAT実装の完全なドキュメントです。GNN (GraphSAGE)実装については`README_GNN.md`を参照してください。
