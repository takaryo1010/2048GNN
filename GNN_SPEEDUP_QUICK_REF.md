# GNN高速化 - クイックリファレンス

## 🚀 主な変更点（3つ）

### 1. バッチ処理の並列化 ⚡
**最も重要な変更**

**変更箇所:** `lzero/model/gnn_utils.py` - `GraphSAGEConv.forward()`

```python
# 変更前（遅い）
for b in range(batch_size):
    out_b = self._forward_single(x[b], edge_index)
    outputs.append(out_b)

# 変更後（高速）
x_flat = x.view(batch_size * num_nodes, feat_dim)
edge_index_batch = torch.cat([edge_index + b*num_nodes for b in range(batch_size)])
# 全バッチを一括処理
```

**効果:** 5-8倍高速化

### 2. LayerNorm化 🔄
**変更箇所:** `lzero/model/gnn_utils.py` - `GraphSAGE`

```python
# 変更前: BatchNorm（transpose必要）
self.bns = nn.ModuleList()
self.bns.append(nn.BatchNorm1d(hidden_dim))
x = x.transpose(1, 2)  # 遅い！
x = self.bns[i](x)
x = x.transpose(1, 2)  # 遅い！

# 変更後: LayerNorm（transposeなし）
self.norms = nn.ModuleList()
self.norms.append(nn.LayerNorm(hidden_dim))
x = self.norms[i](x)  # そのまま処理
```

**効果:** 1.2-1.5倍高速化

### 3. エッジモード選択 🔗
**変更箇所:** `lzero/model/gnn_utils.py` - `GraphBuilder`

```python
# 3つのモードを追加
class GraphBuilder:
    def __init__(self, grid_size=4, edge_mode='sparse'):
        self.edge_mode = edge_mode  # 'adjacent', 'sparse', 'full'
```

| モード | 4x4エッジ | 用途 |
|--------|----------|------|
| adjacent | 48 | 最速 |
| **sparse** | 80 | **推奨** |
| full | 144 | 精度重視 |

## 📝 設定方法

### 4x4グリッド
`zoo/game_2048/config/stochastic_muzero_2048_gnn_config.py`
```python
edge_mode = 'sparse'  # 追加

policy=dict(
    model=dict(
        edge_mode=edge_mode,  # 追加
        ...
    )
)
```

### 3x3グリッド
`zoo/game_2048/config/stochastic_muzero_2048_gnn_3x3_config.py`
```python
edge_mode = 'adjacent'  # 追加

policy=dict(
    model=dict(
        edge_mode=edge_mode,  # 追加
        ...
    )
)
```

## 📊 結果

### ベンチマーク（GPU: RTX 4060）
```
Full Model (batch=256):
- initial_inference: 14.57 ms/batch (17,573 samples/s)
- recurrent_inference: 13.57 ms/batch (18,871 samples/s)
```

### 高速化率
```
変更前: CNNの約10倍遅い
変更後: CNNの約1-2倍遅い
改善: 5-10倍高速化
```

## ✅ テスト方法

```bash
# 基本動作確認
python test_gnn_model_basic.py

# ベンチマーク
python test_gnn_speedup.py
python test_full_model_speed.py

# トレーニング実行
python zoo/game_2048/config/stochastic_muzero_2048_gnn_config.py
```

## 📁 変更ファイル

```
lzero/model/
├── gnn_utils.py                          # メイン変更
└── gnn_stochastic_muzero_model.py        # edge_mode対応

zoo/game_2048/config/
├── stochastic_muzero_2048_gnn_config.py      # 4x4設定
└── stochastic_muzero_2048_gnn_3x3_config.py  # 3x3設定
```

## 🎯 結果

**✅ 約5-10倍の高速化を達成！**

CNNモデルと比較しても実用的な速度（1-2倍遅い程度）で動作可能になりました。
