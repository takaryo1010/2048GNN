# MCTS機能 - 使い方ガイド

## 🎯 概要

**MCTS（Monte Carlo Tree Search）**機能を使用すると、GNNモデルの推論をさらに強化できます。

## 🚀 基本的な使い方

### ポリシーのみ（デフォルト）

```bash
# 通常のポリシーネットワークのみで推論
python gnn_any_size_emulator.py --grid-size 4 --episodes 10
```

**特徴:**
- ✅ 高速（1手あたり 0.01〜0.05秒）
- ✅ リアルタイム実行に適している
- ⚠️ 局所的な判断のみ

### MCTS付き

```bash
# MCTSで推論を強化（50シミュレーション）
python gnn_any_size_emulator.py --grid-size 4 --episodes 10 --use-mcts --num-simulations 50
```

**特徴:**
- ✅ より良いスコア
- ✅ 先読みによる戦略的な判断
- ⚠️ 遅い（1手あたり 0.5〜2秒）

## 📊 シミュレーション回数の選び方

| 回数 | 速度 | 性能 | 用途 |
|-----|------|------|------|
| 10-20 | ⚡️⚡️⚡️ | ⭐️⭐️ | クイックテスト |
| 30-50 | ⚡️⚡️ | ⭐️⭐️⭐️ | **推奨: バランス** |
| 100-150 | ⚡️ | ⭐️⭐️⭐️⭐️ | 高性能 |
| 200+ | 🐌 | ⭐️⭐️⭐️⭐️⭐️ | 最高性能 |

## 💡 実用例

### 例1: クイック評価

```bash
# 3×3盤面で軽量MCTS（20シミュレーション）
python gnn_any_size_emulator.py \
  --grid-size 3 \
  --episodes 5 \
  --use-mcts \
  --num-simulations 20
```

### 例2: バランス型

```bash
# 4×4盤面で標準MCTS（50シミュレーション）
python gnn_any_size_emulator.py \
  --grid-size 4 \
  --episodes 10 \
  --use-mcts \
  --num-simulations 50
```

### 例3: 最高性能

```bash
# 4×4盤面で強力MCTS（150シミュレーション）
python gnn_any_size_emulator.py \
  --grid-size 4 \
  --episodes 3 \
  --use-mcts \
  --num-simulations 150
```

### 例4: GIF作成（MCTS付き）

```bash
# MCTSでゲームプレイを録画
python gnn_any_size_emulator.py \
  --grid-size 4 \
  --episodes 1 \
  --use-mcts \
  --num-simulations 100 \
  --save-gif \
  --gif-path ./mcts_game.gif
```

## 🔬 性能比較の方法

### 比較スクリプトの実行

```bash
# 自動比較テスト
bash compare_mcts.sh
```

### 手動比較

```bash
# ステップ1: ポリシーのみで評価
python gnn_any_size_emulator.py --grid-size 4 --episodes 20

# ステップ2: MCTSで評価
python gnn_any_size_emulator.py --grid-size 4 --episodes 20 --use-mcts --num-simulations 50

# 結果を比較！
```

## 📈 期待される改善

### 3×3盤面

| モード | 平均スコア | 最大タイル |
|--------|-----------|-----------|
| ポリシーのみ | 800-1200 | 32-64 |
| MCTS (30sim) | 1000-1400 | 64-128 |
| MCTS (50sim) | 1100-1500 | 64-128 |

### 4×4盤面

| モード | 平均スコア | 最大タイル |
|--------|-----------|-----------|
| ポリシーのみ | 8000-12000 | 512-1024 |
| MCTS (50sim) | 10000-15000 | 1024-2048 |
| MCTS (100sim) | 12000-18000 | 1024-2048 |

*注: 実際の結果は環境とモデルの学習状態に依存します*

## ⚡ パフォーマンス

### 実行時間（1エピソードあたり）

| 盤面サイズ | ポリシーのみ | MCTS (50sim) | MCTS (100sim) |
|-----------|------------|-------------|--------------|
| 3×3 | 1-3秒 | 10-20秒 | 20-40秒 |
| 4×4 | 5-10秒 | 60-120秒 | 120-240秒 |
| 5×5 | 20-40秒 | 5-10分 | 10-20分 |

## 🎓 MCTSの仕組み

1. **ルートノード作成**: 現在の盤面状態からスタート
2. **シミュレーション**: 複数回のプレイアウトを実行
3. **UCB1選択**: 探索と活用のバランスを取る
4. **バックプロパゲーション**: 結果を親ノードに伝播
5. **最終決定**: 最も訪問されたアクションを選択

## 🔧 高度な設定

### カスタムc_puct値

MCTSの探索度合いを調整したい場合は、コードを編集：

```python
mcts = SimpleMCTS(
    agent=self,
    num_simulations=self.num_simulations,
    device=self.device,
    c_puct=1.0  # デフォルト: 1.0、大きいほど探索的
)
```

## 💡 ベストプラクティス

1. **開発時**: ポリシーのみで高速イテレーション
2. **評価時**: MCTS 50-100シミュレーションで性能測定
3. **デモ時**: MCTS 100-200シミュレーションで最高品質
4. **小さい盤面**: MCTS 20-50シミュレーションで十分
5. **大きい盤面**: MCTS 50-100シミュレーションを推奨

## 🐛 トラブルシューティング

### 遅すぎる

```bash
# シミュレーション回数を減らす
python gnn_any_size_emulator.py --use-mcts --num-simulations 20

# または小さい盤面で試す
python gnn_any_size_emulator.py --grid-size 3 --use-mcts --num-simulations 30
```

### メモリ不足

```bash
# CPUを使用
python gnn_any_size_emulator.py --device cpu --use-mcts --num-simulations 30
```

### スコアが改善しない

- シミュレーション回数を増やす（100-200）
- 盤面サイズを確認（大きい盤面ほど難しい）
- エピソード数を増やして平均を取る

---

**より良い戦略で2048をマスターしよう！🎮🧠**
