# GAT実装 リファクタリング仕様書

**作成日**: 2025-10-27  
**対象**: Graph Attention Network (GAT) 実装の性能改善  
**優先度**: 高（パフォーマンスボトルネック解消）

---

## 📋 目次

1. [現状分析](#現状分析)
2. [パフォーマンスボトルネック](#パフォーマンスボトルネック)
3. [改善提案](#改善提案)
4. [実装優先度](#実装優先度)
5. [詳細仕様](#詳細仕様)
6. [期待効果](#期待効果)

---

## 現状分析

### パフォーマンス測定結果

| 指標 | GAT | CNN | 比率 |
|-----|-----|-----|------|
| 理論演算量 | 901,792 FLOPs | 3,463,782 FLOPs | **0.26x** (1/4) |
| 実行速度 | 7.58 steps/sec | 16.56 steps/sec | **2.18x 遅い** |
| 平均報酬 | 3,392.69 | 4,161.77 | **18.5% 低い** |
| 総実行時間 | 265,606s (73.8h) | 141,570s (39.3h) | **87.6% 長い** |

### 効率ギャップ

```
理論演算量: GAT は CNN の 0.26x （少ない）
実行速度:   GAT は CNN の 2.18x （遅い）
→ 効率損失: 約 8.4倍 (0.26 × 2.18 ≈ 0.57 → 1/0.57 ≈ 1.75倍の非効率)
```

### ボトルネック内訳（推定）

1. **不規則なメモリアクセス**: 50-60% の速度低下
2. **グラフ構築オーバーヘッド**: 20-30% の速度低下
3. **小規模グラフの並列化不足**: 10-20% の速度低下

---

## パフォーマンスボトルネック

### 1. グラフ構築の毎回実行

**問題点**:
```python
# gat_stochastic_muzero_model.py (GATRepresentationNetwork.forward)
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # ❌ 毎回グラフ構築が実行される
    node_features, edge_index = self.graph_builder.obs_to_graph(x)
    node_embeddings = self.gat(node_features, edge_index)
```

**影響**:
- エッジリストは静的（4×4グリッドは常に同じ構造）
- 毎ステップで同じ計算を繰り返している
- CPU-GPU間のデータ転送が頻発

### 2. 非効率なソフトマックス計算

**問題点**:
```python
# gat_utils.py (GraphAttentionConv._edge_softmax)
def _edge_softmax(self, alpha: torch.Tensor, dst: torch.Tensor, 
                 num_nodes: int) -> torch.Tensor:
    # ❌ scatter_reduce を2回実行（max計算とsum計算）
    alpha_max = alpha_max.scatter_reduce(1, dst_expanded, alpha, 
                                        reduce='amax', include_self=False)
    alpha_sum = alpha_sum.scatter_add(1, dst_expanded, alpha_exp)
```

**影響**:
- スキャッター操作は非常に遅い（不規則メモリアクセス）
- エッジごとに独立した計算が必要
- 並列化が困難

### 3. マルチヘッドアテンションの冗長な計算

**問題点**:
```python
# gat_utils.py (GraphAttentionConv.forward)
# ❌ 各ヘッドで独立に線形変換を実行
x_transformed = self.lin(x)  # [B, N, H * D_out]
x_transformed = x_transformed.view(batch_size, num_nodes, H, self.out_dim)
```

**影響**:
- 4つのヘッドで重複した計算
- メモリフットプリントが4倍
- キャッシュ効率が悪い

### 4. バッチ処理の非効率性

**問題点**:
```python
# ❌ エッジごとにバッチ次元を個別に処理
x_src = x_transformed[:, src, :, :]  # [B, E, H, D_out]
x_dst = x_transformed[:, dst, :, :]  # [B, E, H, D_out]
```

**影響**:
- GPUの並列処理能力を活かせない
- 小さいグラフ（16ノード）では特に非効率

### 5. 位置エンコーディングの毎回計算

**問題点**:
```python
# gnn_utils.py (GraphBuilder._get_positional_encoding)
def _get_positional_encoding(self, batch_size: int, device: torch.device):
    # ❌ 毎回同じ位置エンコーディングを計算
    for i in range(self.grid_size):
        for j in range(self.grid_size):
            row_norm = i / (self.grid_size - 1)
            col_norm = j / (self.grid_size - 1)
```

**影響**:
- 静的な情報を動的に計算
- CPUループが遅い

---

## 改善提案

### 優先度A（Critical）: 即座に実装すべき改善

#### A-1. エッジインデックスとデバイス配置の最適化

**目的**: グラフ構築のオーバーヘッドを削減（推定20-30%の高速化）

**変更内容**:
- エッジインデックスを事前にGPUに配置
- 位置エンコーディングをキャッシュ

**実装箇所**: `gnn_utils.py` - `GraphBuilder`

**詳細仕様**:
```python
class GraphBuilder:
    def __init__(self, grid_size: int = 4, include_row_col_edges: bool = True, 
                 edge_mode: str = 'full', device: Optional[torch.device] = None):
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size
        self.edge_mode = edge_mode if include_row_col_edges else 'adjacent'
        
        # ✅ エッジインデックスを事前計算してデバイスに配置
        self.edge_index = self._build_edge_index()
        if device is not None:
            self.edge_index = self.edge_index.to(device)
        
        # ✅ 位置エンコーディングを事前計算
        self._pos_encoding_cache = {}
    
    def obs_to_graph(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = obs.size(0)
        device = obs.device
        
        # ✅ エッジインデックスのデバイス移動を1回だけ
        if self.edge_index.device != device:
            self.edge_index = self.edge_index.to(device)
        
        # ノード特徴抽出
        node_features = obs.flatten(2).transpose(1, 2)
        
        # ✅ 位置エンコーディングをキャッシュから取得
        pos_encoding = self._get_cached_positional_encoding(batch_size, device)
        node_features = torch.cat([node_features, pos_encoding], dim=-1)
        
        return node_features, self.edge_index
    
    def _get_cached_positional_encoding(self, batch_size: int, 
                                       device: torch.device) -> torch.Tensor:
        """キャッシュされた位置エンコーディングを返す"""
        cache_key = (batch_size, device)
        
        if cache_key not in self._pos_encoding_cache:
            # 初回のみ計算
            positions = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    row_norm = i / (self.grid_size - 1) if self.grid_size > 1 else 0.5
                    col_norm = j / (self.grid_size - 1) if self.grid_size > 1 else 0.5
                    positions.append([row_norm, col_norm])
            
            pos_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
            pos_encoding = pos_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            self._pos_encoding_cache[cache_key] = pos_encoding
        
        return self._pos_encoding_cache[cache_key]
```

**期待効果**: 20-30%の高速化

---

#### A-2. ソフトマックス計算の最適化

**目的**: アテンション計算のボトルネック解消（推定15-20%の高速化）

**変更内容**:
- PyTorch Geometric の `softmax` 関数を使用
- カスタム実装を削除

**実装箇所**: `gat_utils.py` - `GraphAttentionConv`

**詳細仕様**:
```python
# 依存関係追加
try:
    from torch_geometric.utils import softmax as pyg_softmax
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

class GraphAttentionConv(nn.Module):
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, feat_dim = x.size()
        H = self.num_heads
        
        # 線形変換
        x_transformed = self.lin(x)
        x_transformed = x_transformed.view(batch_size, num_nodes, H, self.out_dim)
        
        # エッジごとのアテンション計算
        src, dst = edge_index[0], edge_index[1]
        x_src = x_transformed[:, src, :, :]
        x_dst = x_transformed[:, dst, :, :]
        x_edge = torch.cat([x_src, x_dst], dim=-1)
        
        # アテンションスコア
        alpha = (x_edge * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        
        # ✅ PyG の最適化されたソフトマックスを使用
        if HAS_PYG:
            # バッチ処理対応
            alpha_soft_list = []
            for b in range(batch_size):
                for h in range(H):
                    alpha_bh = alpha[b, :, h]  # [E]
                    alpha_soft_bh = pyg_softmax(alpha_bh, dst)
                    alpha_soft_list.append(alpha_soft_bh)
            
            alpha_soft = torch.stack(alpha_soft_list).view(batch_size, -1, H)
        else:
            # フォールバック: カスタム実装
            alpha_soft = self._edge_softmax(alpha, dst, num_nodes)
        
        # ドロップアウト
        alpha_soft = F.dropout(alpha_soft, p=self.dropout, training=self.training)
        
        # メッセージアグリゲーション
        # ... 残りは同じ
```

**期待効果**: 15-20%の高速化

---

#### A-3. 融合カーネル（Fused Operations）の導入

**目的**: メモリアクセスの削減（推定10-15%の高速化）

**変更内容**:
- アテンション計算とメッセージパッシングを融合
- 中間テンソルの生成を削減

**実装箇所**: `gat_utils.py` - `GraphAttentionConv`

**詳細仕様**:
```python
class GraphAttentionConv(nn.Module):
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # ... 前処理 ...
        
        # ✅ 融合カーネル: アテンション計算 + メッセージパッシング
        # 中間テンソル x_edge を保存せず、直接計算
        
        src, dst = edge_index[0], edge_index[1]
        num_edges = src.size(0)
        
        # 一時バッファを再利用
        out = torch.zeros(batch_size, num_nodes, H, self.out_dim,
                         dtype=x.dtype, device=x.device)
        
        # ヘッドごとにループ（メモリ効率優先）
        for h in range(H):
            # 各ヘッドの線形変換
            W_h = self.lin.weight.view(H, self.out_dim, -1)[h]  # [D_out, D_in]
            x_h = torch.matmul(x, W_h.t())  # [B, N, D_out]
            
            # アテンションスコア計算
            a_h = self.att[0, h, :]  # [2*D_out]
            a_src, a_dst = a_h[:self.out_dim], a_h[self.out_dim:]
            
            alpha = (x_h[:, src, :] * a_src).sum(dim=-1) + \
                    (x_h[:, dst, :] * a_dst).sum(dim=-1)
            alpha = F.leaky_relu(alpha, 0.2)
            
            # ソフトマックス
            alpha_soft = pyg_softmax(alpha.view(-1), dst.repeat(batch_size))
            alpha_soft = alpha_soft.view(batch_size, num_edges)
            
            # メッセージアグリゲーション（インプレース）
            for b in range(batch_size):
                out[b, :, h, :].index_add_(
                    0, dst, 
                    alpha_soft[b, :, None] * x_h[b, src, :]
                )
        
        # ... 後処理 ...
```

**期待効果**: 10-15%の高速化

---

### 優先度B（High）: 重要だが後回しでも可

#### B-1. スパースアテンションの導入

**目的**: 計算量の削減（推定5-10%の高速化）

**変更内容**:
- エッジモードを 'adjacent' に固定（88エッジ → 56エッジ）
- 不要なエッジを削除

**実装箇所**: `gat_stochastic_muzero_model.py`

**詳細仕様**:
```python
class GATRepresentationNetwork(nn.Module):
    def __init__(self, ...):
        # ✅ 'adjacent' モードに固定（最速）
        self.graph_builder = GraphBuilder(
            grid_size, 
            include_row_col_edges=False,  # adjacent モード
            edge_mode='adjacent'
        )
```

**期待効果**: 5-10%の高速化、メモリ使用量30%削減

---

#### B-2. マルチヘッドアテンションのバッチ処理最適化

**目的**: GPU並列度の向上（推定5-10%の高速化）

**変更内容**:
- ヘッドをバッチ次元に統合
- 一度に全ヘッドを計算

**実装箇所**: `gat_utils.py` - `GraphAttentionConv`

**詳細仕様**:
```python
class GraphAttentionConv(nn.Module):
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, feat_dim = x.size()
        H = self.num_heads
        
        # ✅ バッチとヘッドを統合: [B, N, D] -> [B*H, N, D_out]
        x_transformed = self.lin(x)  # [B, N, H*D_out]
        x_transformed = x_transformed.view(batch_size * H, num_nodes, self.out_dim)
        
        # エッジインデックスをバッチ×ヘッド分複製
        edge_index_expanded = edge_index.repeat(1, batch_size * H)
        for i in range(batch_size * H):
            offset = (i // H) * num_nodes
            edge_index_expanded[:, i*edge_index.size(1):(i+1)*edge_index.size(1)] += offset
        
        # 一度に全バッチ×全ヘッドを処理
        # ... アテンション計算 ...
        
        # 結果を再形成: [B*H, N, D_out] -> [B, N, H, D_out]
        out = out.view(batch_size, H, num_nodes, self.out_dim)
        out = out.transpose(1, 2)  # [B, N, H, D_out]
```

**期待効果**: 5-10%の高速化

---

#### B-3. レイヤー正規化の最適化

**目的**: 正規化のオーバーヘッド削減（推定3-5%の高速化）

**変更内容**:
- グループ正規化を使用（LayerNormより高速）
- または正規化を削除（実験的）

**実装箇所**: `gat_utils.py` - `GraphAttention`

**詳細仕様**:
```python
class GraphAttention(nn.Module):
    def __init__(self, ..., norm_type: str = 'layer'):
        """
        norm_type: 'layer', 'group', 'none'
        """
        for i in range(num_layers):
            if norm_type == 'layer':
                self.norms.append(nn.LayerNorm(hidden_dim))
            elif norm_type == 'group':
                # ✅ GroupNorm（より高速）
                num_groups = min(32, hidden_dim // 4)
                self.norms.append(nn.GroupNorm(num_groups, hidden_dim))
            else:
                self.norms.append(nn.Identity())
```

**期待効果**: 3-5%の高速化

---

### 優先度C（Medium）: パフォーマンス向上の可能性

#### C-1. Mixed Precision Training

**目的**: メモリ帯域幅とFLOPsの削減（推定10-20%の高速化）

**変更内容**:
- FP16/BF16を使用
- Automatic Mixed Precision (AMP) の有効化

**実装箇所**: トレーニングスクリプト

**詳細仕様**:
```python
# train_gat_stochastic.py
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        # ✅ FP16で計算
        output = model(batch)
        loss = criterion(output, target)
    
    # バックワード（FP32スケーリング）
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**期待効果**: 10-20%の高速化、メモリ使用量50%削減

---

#### C-2. カスタムCUDAカーネルの実装

**目的**: GPUカーネルの最適化（推定20-30%の高速化）

**変更内容**:
- アテンション計算用のカスタムCUDAカーネル
- 融合されたsoftmax+scatter操作

**実装箇所**: 新規ファイル `gat_cuda_kernels.cu`

**注意**: C++/CUDA の専門知識が必要、メンテナンスコストが高い

**期待効果**: 20-30%の高速化（実装コスト大）

---

#### C-3. グラフのプリプロセッシング

**目的**: 実行時のグラフ構築を完全に排除（推定15-25%の高速化）

**変更内容**:
- データローダーでグラフを事前生成
- モデルはグラフを受け取るだけ

**実装箇所**: データローダー

**詳細仕様**:
```python
class Game2048GraphDataset:
    def __getitem__(self, idx):
        obs = self.observations[idx]
        
        # ✅ データローダーでグラフ化
        node_features, edge_index = self.graph_builder.obs_to_graph(obs)
        
        return {
            'obs': obs,
            'node_features': node_features,
            'edge_index': edge_index,
            'action': self.actions[idx],
            'reward': self.rewards[idx]
        }
```

**期待効果**: 15-25%の高速化

---

## 実装優先度

### フェーズ1（即座に実装）: 優先度A

**実装順序**:
1. A-1: エッジインデックスとデバイス配置の最適化（2時間）
2. A-2: ソフトマックス計算の最適化（3時間）
3. A-3: 融合カーネルの導入（4時間）

**期待効果**: 合計 45-65%の高速化  
**実装時間**: 9時間  
**リスク**: 低（既存機能の最適化のみ）

---

### フェーズ2（1週間以内）: 優先度B

**実装順序**:
1. B-1: スパースアテンションの導入（1時間）
2. B-2: マルチヘッドアテンションの最適化（4時間）
3. B-3: レイヤー正規化の最適化（2時間）

**期待効果**: 追加で 13-25%の高速化  
**実装時間**: 7時間  
**リスク**: 中（一部の性能劣化の可能性）

---

### フェーズ3（必要に応じて）: 優先度C

**実装順序**:
1. C-1: Mixed Precision Training（3時間）
2. C-3: グラフのプリプロセッシング（5時間）
3. C-2: カスタムCUDAカーネル（20時間以上）

**期待効果**: 追加で 25-55%の高速化  
**実装時間**: 28時間以上  
**リスク**: 高（実装の複雑さ、メンテナンスコスト）

---

## 詳細仕様

### ファイル構成

```
LightZero/lzero/model/
├── gat_utils.py              # 修正対象（A-2, A-3, B-2, B-3）
├── gnn_utils.py              # 修正対象（A-1）
├── gat_stochastic_muzero_model.py  # 修正対象（B-1）
└── gat_cuda_kernels.cu       # 新規（C-2, オプション）

2048GNN/
├── train_gat_stochastic.py   # 修正対象（C-1）
└── configs/
    └── gat_optimized_config.py  # 新規設定ファイル
```

---

### テスト計画

#### 単体テスト

```python
# tests/test_gat_optimizations.py

def test_edge_index_caching():
    """A-1: エッジインデックスのキャッシングをテスト"""
    graph_builder = GraphBuilder(grid_size=4, device='cuda')
    obs = torch.randn(32, 16, 4, 4, device='cuda')
    
    # 初回
    import time
    start = time.time()
    _, edge_index1 = graph_builder.obs_to_graph(obs)
    time1 = time.time() - start
    
    # 2回目（キャッシュ使用）
    start = time.time()
    _, edge_index2 = graph_builder.obs_to_graph(obs)
    time2 = time.time() - start
    
    assert torch.equal(edge_index1, edge_index2)
    assert time2 < time1 * 0.5  # 50%以上高速化


def test_softmax_equivalence():
    """A-2: ソフトマックス最適化の等価性をテスト"""
    conv = GraphAttentionConv(18, 128, num_heads=4)
    x = torch.randn(32, 16, 18)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
    
    # オリジナル実装
    out1 = conv.forward_original(x, edge_index)
    
    # 最適化実装
    out2 = conv.forward(x, edge_index)
    
    assert torch.allclose(out1, out2, rtol=1e-4)


def test_memory_usage():
    """メモリ使用量のテスト"""
    import torch.cuda as cuda
    
    model = GATStochasticMuZeroModel(...)
    obs = torch.randn(256, 16, 4, 4, device='cuda')
    
    cuda.reset_peak_memory_stats()
    output = model(obs)
    peak_memory = cuda.max_memory_allocated()
    
    assert peak_memory < 2 * 1024**3  # 2GB以下
```

#### 統合テスト

```python
def test_training_speed():
    """トレーニング速度のベンチマーク"""
    # オリジナル実装
    model_orig = GATStochasticMuZeroModel(...)
    time_orig = benchmark_training(model_orig, num_steps=1000)
    
    # 最適化実装
    model_opt = GATStochasticMuZeroModelOptimized(...)
    time_opt = benchmark_training(model_opt, num_steps=1000)
    
    speedup = time_orig / time_opt
    assert speedup > 1.4  # 最低40%の高速化
```

---

### ロールバック計画

各最適化は独立して実装され、問題が発生した場合は個別に無効化可能：

```python
# gat_optimized_config.py

optimization_flags = dict(
    use_edge_caching=True,        # A-1
    use_pyg_softmax=True,         # A-2
    use_fused_kernels=True,       # A-3
    edge_mode='adjacent',         # B-1
    use_batched_heads=True,       # B-2
    norm_type='group',            # B-3
    use_amp=False,                # C-1（デフォルト無効）
    preprocess_graphs=False,      # C-3（デフォルト無効）
)
```

---

## 期待効果

### 総合的な性能改善予測

| フェーズ | 実装内容 | 高速化 | 累積高速化 | 実装時間 |
|---------|---------|--------|-----------|---------|
| 現状 | - | 1.00x | 1.00x | - |
| フェーズ1 | A-1, A-2, A-3 | 1.45-1.65x | 1.45-1.65x | 9h |
| フェーズ2 | B-1, B-2, B-3 | 1.13-1.25x | 1.64-2.06x | 7h |
| フェーズ3 | C-1, C-3 | 1.25-1.55x | 2.05-3.19x | 8h |

### 最良シナリオ（全実装）

```
現状:      7.58 steps/sec
フェーズ1: 11.0-12.5 steps/sec  (+45-65%)
フェーズ2: 12.4-15.5 steps/sec  (+64-106%)
フェーズ3: 15.5-24.2 steps/sec  (+105-219%)
```

**CNNとの比較**:
- 現状: GAT 7.58 vs CNN 16.56 → **2.18倍遅い**
- 最適化後: GAT 15.5-24.2 vs CNN 16.56 → **0.68-1.46倍**

---

### 投資対効果（ROI）

| フェーズ | 実装時間 | 高速化 | ROI |
|---------|---------|--------|-----|
| フェーズ1 | 9時間 | 45-65% | **高い** |
| フェーズ2 | 7時間 | 13-25% | 中 |
| フェーズ3 | 28時間 | 25-55% | 低い |

**推奨**: フェーズ1のみ実装し、その後の効果を測定してからフェーズ2に進む

---

## 実装チェックリスト

### フェーズ1（優先度A）

- [ ] A-1: GraphBuilder のデバイス配置最適化
  - [ ] エッジインデックスのGPU事前配置
  - [ ] 位置エンコーディングのキャッシング
  - [ ] 単体テスト作成
  - [ ] ベンチマーク実行

- [ ] A-2: ソフトマックス最適化
  - [ ] PyTorch Geometric の softmax 統合
  - [ ] フォールバック実装の維持
  - [ ] 数値精度の検証
  - [ ] パフォーマンステスト

- [ ] A-3: 融合カーネル
  - [ ] アテンション+メッセージパッシング融合
  - [ ] 中間テンソル削減
  - [ ] メモリ使用量の測定
  - [ ] 正確性の検証

### フェーズ2（優先度B）

- [ ] B-1: スパースアテンション
  - [ ] エッジモードの変更
  - [ ] パフォーマンス比較
  - [ ] 精度への影響評価

- [ ] B-2: マルチヘッドバッチ処理
  - [ ] バッチ×ヘッド統合
  - [ ] GPU並列度の測定
  - [ ] スループット比較

- [ ] B-3: 正規化最適化
  - [ ] GroupNorm の実装
  - [ ] 収束性の検証
  - [ ] 速度比較

### フェーズ3（優先度C）

- [ ] C-1: Mixed Precision
  - [ ] AMP の有効化
  - [ ] 数値安定性の確認
  - [ ] メモリ削減の測定

- [ ] C-3: グラフプリプロセッシング
  - [ ] データローダー改修
  - [ ] キャッシュ機構の実装
  - [ ] I/Oボトルネックの確認

---

## まとめ

### 現実的な目標

**フェーズ1（9時間の実装）で期待される効果**:
- 実行速度: 7.58 → 11-12.5 steps/sec (**45-65%高速化**)
- トレーニング時間: 73.8h → 50-59h (**20-32%削減**)
- CNN比: 2.18倍遅い → 1.3-1.5倍遅い

**結論**:
- GATは構造的にCNNより不利（小規模グリッドでは特に）
- 最適化でギャップを縮められるが、完全には追いつかない
- **推奨**: 2048ゲームでは引き続きCNNを使用し、GATは研究目的に限定

---

**作成者**: AI Assistant  
**レビュー**: 未  
**承認**: 未  
**最終更新**: 2025-10-27
