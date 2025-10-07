# Stochastic MuZero の CNN -> GNN 置き換え（日本語プロンプト）

目的: Stochastic MuZero を 2048 ゲームに適用する際、従来の CNN ベース表現を GNN に置き換えるための AI/エンジニア向けプロンプト。Hex の GraphAra（AlphaZero 風）事例を参照し、実装上の設計指針、手順、契約、注意点、擬似コード、設定例をまとめる。

## 1. プロジェクトの目的と背景
- 目標: 2048 に対して MuZero の表現部・予測ヘッドを GNN に置き換え、長距離依存をより良く扱い一般化性能を改善する。
- 背景: CNN（ResNet 等）は局所的な畳み込みで特徴を学習するが、非局所的・リレーショナルな依存関係（例: あるタイルがスライドして連鎖を作るような関係）を効率的に扱えない。
- 参考: Hex における GraphAra の設計思想（ボードをグラフとして扱い、GNN で局所・非局所の関係を学習）を MuZero の Representation/Prediction/Dynamics に適用する。

## 2. 最重要前提: グラフ表現 G=(V,E) の設計
- ノード (V): 各グリッドセルを 1 ノードとして表現。状態に応じてノード属性にタイル値（例: log2(value) の正規化）、空セルフラグなどを含める。
- エッジ (E): エッジは設計次第でタスク固有にする。候補:
  - 格子隣接（上下左右）: 基本的なローカル伝播を保証。
  - 行/列ペア: 同じ行・列にあるノードを完全グラフ的に結ぶことでスライドやマージ依存を扱う。
  - 値類似エッジ: 同じ値や合致しやすいペアを動的に繋ぐ（計算上高コストなので注意）。
  - グローバルタイプの仮想ノード: 全ノードを集約するためのハブノードを追加して長距離伝播を効率化。
- ノード特徴量の例:
  - one-hot またはスカラーで表す log2(value)
  - 空/埋有フラグ
  - 位置情報（行・列のインデックス、二次元座標の正規化値）
- エッジ属性（必要な場合）:
  - 方向（e.g., from->to）、距離（マンハッタン距離）、行/列同一フラグ

## 3. 置き換えるべき MuZero コンポーネント
全体のアイデア: Representation Network（状態 s の埋め込み生成）を GNN に。Prediction Network の Policy/Value/Reward ヘッドも GNN 埋め込みに基づいて再実装する。Dynamics Network（状態遷移）の表現も GNN ベースのメッセージパッシングで置き換え可能。

### 3.1 Representation Network (特徴抽出)
- 置き換え対象: CNN/ResNet ベースのブロック
- 推奨アーキテクチャ: GraphSAGE、GAT（注意: GAT 実装で問題がある場合は GraphSAGE で十分堅牢）、または GIN（Graph Isomorphism Network）
- 入力: グラフ G とノード特徴量
- 出力: 各ノードの埋め込みベクトル E_node
- 層の設計例: 3-6 層の GraphSAGE ブロック、各層で BatchNorm/LayerNorm と残差接続を検討

### 3.2 Prediction Network - Value ヘッド
- 入力: E_node
- 処理:
  1. ノード埋め込み全体に対して複数の対称集約（mean, max, sum）を計算
  2. 集約ベクトルを連結して単一ベクトルにする
  3. MLP (2-3 層) を通してスカラー値 V(s) を出力
- ロス/スケーリング: MuZero の価値スケールに合わせた正規化を行う（ターゲット正規化など）

### 3.3 Prediction Network - Policy ヘッド
- 選択肢 A (行動をノードに対応付ける場合): 各ノードに対してスカラーのスコアを計算し、行動空間（スライド/方向など）にマッピングして softmax を適用
- 選択肢 B (行動をグローバルに扱う場合): ノード埋め込みを集約して MLP から行動次元数のスコアを出す
- 実装推奨: 2048 の行動は小さい（4方向）。policy head は lightweight にし、GraphSAGE を policy 専用に微調整したブランチを用意する

### 3.4 Dynamics Network
- 役割: 現在の埋め込みと行動から次状態の埋め込みを予測する
- GNN 化: 現在のノード埋め込みに行動の埋め込み（行動ワンホットをノードにブロードキャスト、または特定ノードにアタッチ）を加え、数層のメッセージパッシングを行う

## 4. 実装時の「小さな契約」(Contract)
- 入力/出力の形状:
  - 入力: ノード数 N（固定: 16 for 4x4）、ノード特徴量次元 D_in
  - 出力: ノード埋め込み E_node (N x D_out)、Value スカラー、Policy ベクトル（size A）
- エラー/例外モード:
  - グラフが破損（ノード欠落など）した際は例外を投げる
  - 可変ノード数をサポートする場合、集約操作を必須にする
- 性能/速度目標:
  - 1 forward pass が既存 CNN 実装の 2x 以内を目指す（ハードウェア次第）

## 5. エッジケースと注意点
- 可変ノード数: 2048 の場合固定だが、別のボードサイズに対応するなら可変ノード数対応を検討
- 行動表現の整合性: 行動をどのようにノードにマップするかを明確にしないと、Policy ヘッドが意図した通りに学習しない
- GAT 実装課題: 提示されたように既存 GAT ファイルは不安定なら GraphSAGE を先に採用する

## 6. 擬似コード（PyTorch スタイル）
# ...existing code...

### 6.1 GraphSAGE ブロック（簡易実装）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGEConv(nn.Module):
  def __init__(self, in_dim, out_dim, agg='mean'):
    super().__init__()
    self.agg = agg
    self.lin = nn.Linear(in_dim*2, out_dim)

  def forward(self, x, edge_index):
    # x: [N, in_dim]
    # edge_index: [2, E] (source, target)
    src, dst = edge_index
    # aggregate neighbor features
    if self.agg == 'mean':
      deg = torch.zeros(x.size(0), device=x.device).index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
      neigh = torch.zeros_like(x).index_add_(0, dst, x[src])
      neigh = neigh / (deg.unsqueeze(-1).clamp(min=1.0))
    else:
      # fallback to sum
      neigh = torch.zeros_like(x).index_add_(0, dst, x[src])

    h = torch.cat([x, neigh], dim=-1)
    return F.relu(self.lin(h))


class GraphSAGE(nn.Module):
  def __init__(self, in_dim, hidden_dim, num_layers=3):
    super().__init__()
    layers = []
    dims = [in_dim] + [hidden_dim] * num_layers
    for i in range(num_layers):
      layers.append(GraphSAGEConv(dims[i], dims[i+1]))
    self.layers = nn.ModuleList(layers)

  def forward(self, x, edge_index):
    for conv in self.layers:
      x = conv(x, edge_index)
    return x
```

### 6.2 Value ヘッド（ノード集約 -> MLP）

```python
class ValueHead(nn.Module):
  def __init__(self, node_dim, hidden=128):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(node_dim*3, hidden),
      nn.ReLU(),
      nn.Linear(hidden, hidden//2),
      nn.ReLU(),
      nn.Linear(hidden//2, 1)
    )

  def forward(self, node_emb):
    # node_emb: [N, D]
    mean = node_emb.mean(dim=0)
    maxv, _ = node_emb.max(dim=0)
    sumv = node_emb.sum(dim=0)
    cat = torch.cat([mean, maxv, sumv], dim=-1)
    return self.mlp(cat).squeeze(-1)
```

### 6.3 Policy ヘッド（ノード->行動）

```python
class PolicyHead(nn.Module):
  def __init__(self, node_dim, action_dim=4, hidden=128):
    super().__init__()
    # 選択肢: ノードごとにスコア -> 行動へマッピング
    self.node_to_action = nn.Sequential(
      nn.Linear(node_dim, hidden),
      nn.ReLU(),
      nn.Linear(hidden, action_dim)
    )

  def forward(self, node_emb):
    # node_emb: [N, D]
    per_node_scores = self.node_to_action(node_emb)  # [N, A]
    # 合成戦略: ノード毎のスコアを行動次元で集約して softmax
    agg = per_node_scores.mean(dim=0)  # [A]
    return F.log_softmax(agg, dim=-1)
```

### 6.4 Dynamics の簡易案

```
# 現在のノード埋め込みと行動を結合して数ステップ GNN を回す
def dynamics_step(node_emb, action_id, gnn_module):
  # action_id: scalar int
  A = one_hot(action_id, num_actions)
  # ブロードキャストして各ノードに付与
  a_feat = A.unsqueeze(0).expand(node_emb.size(0), -1)
  x = torch.cat([node_emb, a_feat], dim=-1)
  return gnn_module(x, edge_index)
```

## 7. 設定ファイル例とハイパーパラメータ
- GNN 層数: 3
- 埋め込み次元: 128
- 学習率: 1e-3（スケジューラを推奨）
- バッチサイズ: 64
- 正則化: weight decay + dropout (0.1)

### 7.1 YAML 例

```yaml
model:
  type: GraphSAGE_MuZero
  node_dim: 128
  gnn_layers: 3

train:
  lr: 1e-3
  batch_size: 64
  weight_decay: 1e-4

policy:
  action_dim: 4

``` 

## 8. 期待される効果
- 長距離依存を扱えるため、2048 の戦略的な合併・連鎖形成に有利
- CNN より過学習が減り、検証セットへの一般化が向上する見込み

---
作業済: まずプロンプト文書を作成しました。次に、実装例（GraphSAGE モジュール、ヘッドの擬似コード、config の YAML 例）を追加して、2 番の todo を完了します。

## 参考リンク（リポジトリ内）
- `LightZero/README.md` - LightZero の全体的な使い方、インストール、クイックスタート
- `LightZero/docs/source/tutorials/config/config.md` - 設定ファイルの書き方と例
- `LightZero/zoo/README.md` - zoo 配下のモデル/設定の管理方法
- `LightZero/GAT_INSTALLATION.md` - GAT に関する既存ドキュメント（不安定なら参照のみ）

これらのファイルは本リポジトリ内にあります。実装を進める際は、`Observation -> GraphBuilder` のラッパーを `LightZero/zoo` の既存フォーマットに従って追加してください。