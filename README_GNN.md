# GNN版 Stochastic MuZero for 2048

このプロジェクトは、2048ゲームにおいて従来のCNNベースのStochastic MuZeroをGNN (Graph Neural Network) ベースに置き換えたものです。GraphSAGEアーキテクチャを使用して、グリッド状態をグラフ構造として処理し、長距離依存関係をより効率的に捕捉します。

## 🎯 プロジェクトの目的

CNNの課題:
- 局所的な畳み込みでは非局所的な依存関係（例: 行全体のスライド、対角線上の連鎖）を効率的に扱えない
- 訓練データへの過剰適合の傾向

GNNの利点:
- グラフ構造により、タスク固有のリレーショナルな情報を明示的にエンコード
- 行・列内の全タイルを直接接続することで長距離依存を効率的に学習
- より良い一般化性能と検証データへの適応

## 📁 ファイル構成

```
2048GNN/
├── LightZero/
│   ├── lzero/
│   │   └── model/
│   │       ├── gnn_utils.py                    # GraphBuilder, GraphSAGE実装
│   │       └── gnn_stochastic_muzero_model.py  # GNN版MuZeroモデル
│   └── zoo/
│       └── game_2048/
│           └── config/
│               └── stochastic_muzero_2048_gnn_config.py  # GNN設定
├── test_gnn_model.py                           # 単体テスト
├── GNN_MIGRATION_PROMPT_JA.md                  # 実装ガイド（日本語）
└── README_GNN.md                               # このファイル
```

## 🏗️ アーキテクチャ概要

### グラフ表現 (Graph Representation)

4x4グリッド → 16ノードのグラフ

**ノード特徴量:**
- One-hot エンコードされたタイル値（16次元）
- 位置エンコーディング（行・列の正規化座標、2次元）
- 合計: 18次元

**エッジ構造:**
1. **隣接エッジ**: 上下左右の4方向で隣接セルを双方向接続
2. **行エッジ**: 同じ行内の全セルを完全グラフで接続
3. **列エッジ**: 同じ列内の全セルを完全グラフで接続

→ 合計144エッジ（双方向）で長距離依存を効率的に伝播

### ネットワークコンポーネント

#### 1. Representation Network (GNNRepresentationNetwork)
- **役割**: 観測 → 潜在状態の埋め込み
- **実装**: 3層GraphSAGE
- **入出力**:
  - 入力: [B, 16, 4, 4] (観測)
  - 出力: [B, 128, 4, 4] (潜在状態)

#### 2. Prediction Network (GNNPredictionNetwork)
- **役割**: 潜在状態 → Value & Policy
- **Value Head**: ノード集約（mean/max/sum連結） → MLP → スカラー値
- **Policy Head**: ノード集約 → MLP → 行動確率（4方向）

#### 3. Dynamics Network (GNNDynamicsNetwork)
- **役割**: 潜在状態 + 行動 → 次状態 + 報酬
- **実装**: 行動エンコーディングをノードにブロードキャスト → GraphSAGE → 次状態予測
- **報酬予測**: ノード集約 → MLP → 報酬値

#### 4. Afterstate Networks
- Stochastic MuZero特有のチャンス遷移を処理
- Dynamics/Predictionと同様の構造で実装

## 🚀 使い方

### 1. テスト実行（動作確認）

```bash
cd /opendilab/2048GNN
python test_gnn_model.py
```

**テスト内容:**
- GraphBuilderの動作確認
- Forward passの形状チェック
- 勾配の伝播確認
- CUDA互換性テスト

### 2. 学習の実行

```bash
cd /opendilab/2048GNN/LightZero
python zoo/game_2048/config/stochastic_muzero_2048_gnn_config.py
```

**学習設定:**
- GNN層数: 3
- 隠れ次元: 128
- バッチサイズ: 512
- シミュレーション回数: 100
- 学習率: 0.003

### 3. 設定のカスタマイズ

`stochastic_muzero_2048_gnn_config.py` を編集:

```python
# GNNハイパーパラメータ
num_gnn_layers = 3              # GNN層数
gnn_hidden_dim = 128            # 隠れ次元
include_row_col_edges = True    # 行/列エッジを含める
gnn_dropout = 0.0               # ドロップアウト率

# 学習パラメータ
batch_size = 512
learning_rate = 0.003
num_simulations = 100
```

## 🧪 実験・比較

### CNN vs GNN の比較方法

1. **CNNベースライン実行**:
```bash
python zoo/game_2048/config/stochastic_muzero_2048_config.py
```

2. **GNN実行**:
```bash
python zoo/game_2048/config/stochastic_muzero_2048_gnn_config.py
```

3. **比較指標**:
- 最大タイル値の分布
- エピソードごとのスコア
- 検証データでの一般化性能
- 訓練速度（iterations/sec）

## 📊 期待される効果

参考: Hex における GraphAra の結果

- **エラー率低減**: 長距離依存タスクで顕著な精度向上
- **一般化性能**: 検証データへのより良い適応
- **過学習軽減**: CNN と比較して訓練/検証のギャップが縮小

2048 での期待:
- 戦略的な合併パターン（例: 角に大きいタイルを配置）の学習向上
- 行全体をスライドする判断の精度向上

## 🔧 トラブルシューティング

### メモリ不足エラー

```python
# 設定で調整
batch_size = 256  # 512から減らす
gnn_hidden_dim = 64  # 128から減らす
```

### 学習が遅い

```python
# 行/列エッジを無効化（エッジ数削減）
include_row_col_edges = False

# または層数を減らす
num_gnn_layers = 2
```

### 勾配爆発・消失

```python
# ドロップアウト追加
gnn_dropout = 0.1

# 学習率を下げる
learning_rate = 0.001
```

## 📚 関連ドキュメント

- `GNN_MIGRATION_PROMPT_JA.md` - 実装の理論的背景とプロンプト
- `LightZero/README.md` - LightZeroフレームワークの使い方
- `LightZero/docs/` - 設定ファイルの詳細ドキュメント

## 🔬 今後の拡張

1. **注意機構の追加**: GraphSAGE → GAT (Graph Attention Network)
2. **グローバルノード**: すべてのノードを集約する仮想ノードの追加
3. **動的エッジ**: タイル値に基づいて動的にエッジを構築
4. **マルチスケールGNN**: 異なる解像度でグラフを構築

## 📝 引用

本実装は以下の論文の考え方を参考にしています:

- GraphAra (Hex game): "Graph Neural Networks for Board Game Representation"
- Stochastic MuZero: https://openreview.net/pdf?id=X6D9bAHhBQ1

## ✅ 実装完了チェックリスト

- [x] GraphBuilder実装
- [x] GraphSAGE層実装
- [x] GNN Representation Network
- [x] GNN Prediction Network (Value/Policy heads)
- [x] GNN Dynamics Network
- [x] Afterstate networks
- [x] 設定ファイル作成
- [x] 単体テスト実装・実行
- [x] ドキュメント作成

## 🙏 謝辞

- LightZero フレームワーク
- DI-engine
- PyTorch Geometric (設計参考)

---

**作成日**: 2025-10-07  
**バージョン**: 1.0  
**ライセンス**: Apache 2.0
