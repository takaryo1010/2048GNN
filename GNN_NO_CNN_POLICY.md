# GNNモデル - CNN使用禁止ポリシー

## 概要
このプロジェクトのGNNモデル（`GNNStochasticMuZeroModel`）は、従来のCNNベースの
アーキテクチャを完全にGraph Neural Network (GNN)に置き換えています。

## CNN使用のポリシー

### ✅ 許可される場所
- **chance_encoderのみ**: チャンスノードのエンコーディングに使用されるCNNは許可
  - `chance_encoder.encoder.conv1`
  - `chance_encoder.encoder.conv2`

### ❌ 禁止される場所
以下のコンポーネントではCNNレイヤーの使用は**完全に禁止**:
- `representation_network`: 観測を潜在状態に変換（GNN使用）
- `dynamics_network`: 状態遷移のモデル化（GNN使用）
- `afterstate_dynamics_network`: afterstate遷移（GNN使用）
- `prediction_network`: 価値・方策予測（GNN集約使用）

### 使用されるGNNコンポーネント
- `GraphBuilder`: グリッド観測をグラフ構造に変換
- `GraphSAGE`: グラフ畳み込みネットワーク
- `GraphSAGEConv`: メッセージパッシング層

## アーキテクチャの違い

### CNNモデル（従来）
```
観測 [B,16,4,4] 
  → Conv2d 
  → ResBlock × N 
  → 潜在状態 [B,128,4,4]
```

### GNNモデル（現在）
```
観測 [B,16,4,4] 
  → GraphBuilder (グラフ化)
  → ノード特徴 [B,16,18] + エッジ [2,80]
  → GraphSAGE × 3
  → ノード埋め込み [B,16,128]
  → グリッド再構成
  → 潜在状態 [B,128,4,4]
```

## 利点
1. **パラメータ効率**: CNNの約1/5のパラメータ
2. **明示的なグラフ構造**: エッジで情報伝播を制御
3. **スケーラビリティ**: より大きなグリッドに対応可能
4. **柔軟性**: エッジモードを変更可能（adjacent, sparse, full）

## 検証方法
```python
from validate_no_cnn import validate_no_cnn_usage, validate_gnn_components

model = GNNStochasticMuZeroModel(**config)
validate_no_cnn_usage(model)
validate_gnn_components(model)
```

## エラーハンドリング
GNN部分で誤ってCNNレイヤーを使用しようとすると、
初期化時に`RuntimeError`が発生します。

---
生成日: 2025-10-09
