# GNN 2048 汎用サイズエミュレータ - クイックスタート

4×4で学習したGNNモデルを任意のサイズで即座に実行できます！

## 🚀 30秒で開始

```bash
# 基本的な実行（4×4、10エピソード）
python gnn_any_size_emulator.py

# 3×3盤面で実行
python gnn_any_size_emulator.py --grid-size 3

# 5×5盤面で実行
python gnn_any_size_emulator.py --grid-size 5

# リアルタイム描画付きで実行
python gnn_any_size_emulator.py --grid-size 4 --episodes 3 --render

# MCTSで推論を強化（より良い性能）
python gnn_any_size_emulator.py --grid-size 4 --use-mcts --num-simulations 50
```

## 📊 出力例

```
============================================================
統計情報 (盤面サイズ: 3×3)
============================================================
平均スコア:     1166.67 ± 50.84
最高スコア:     1232
最低スコア:     1108
平均最大タイル: 32
平均手数:       42.7

最大タイル達成回数:
    32: 3回
============================================================
```

## 🎮 サポートする盤面サイズ

| サイズ | スコア目安 | 最大タイル | 実行速度 |
|--------|-----------|-----------|---------|
| 3×3 | 600-1200 | 32-64 | 超高速 |
| 4×4 | 8000-15000 | 512-1024 | 高速 |
| 5×5 | 40000-80000 | 512-2048 | 中速 |
| 6×6 | 80000-150000 | 1024-4096 | 中速 |
| 7×7 | 150000+ | 2048+ | 低速 |

## 💡 よく使うコマンド

### 複数サイズで比較

```bash
# 各サイズで10エピソード実行
for size in 3 4 5 6; do
  echo "Testing ${size}x${size}..."
  python gnn_any_size_emulator.py --grid-size $size --episodes 10
done
```

### GIF動画を作成

```bash
python gnn_any_size_emulator.py \
  --grid-size 4 \
  --episodes 1 \
  --save-gif \
  --gif-path ./my_game.gif
```

### デモスクリプトを実行

```bash
# 3×3, 4×4, 5×5, 6×6 を自動テスト
bash demo_any_size.sh
```

## 🔧 カスタマイズ

### 独自のモデルを使用

```bash
python gnn_any_size_emulator.py \
  --model-path /path/to/your/model.pth.tar \
  --grid-size 4 \
  --episodes 20
```

### CPUで実行

```bash
python gnn_any_size_emulator.py \
  --device cpu \
  --grid-size 5 \
  --episodes 10
```

## 📖 詳細ドキュメント

詳しい使い方は **GNN_ANY_SIZE_EMULATOR_README.md** を参照してください。

## 🎯 主な機能

✅ LightZero GUIに依存しない独立動作  
✅ 3×3〜8×8の任意サイズに対応  
✅ **MCTSによる推論強化（新機能！）**  
✅ リアルタイム描画とGIF保存  
✅ 詳細な統計情報の自動集計  
✅ カスタムモデルの簡単な読み込み  

## 🚀 MCTS機能

### MCTSとは？

Monte Carlo Tree Search（MCTS）を使用すると、モデルがより深く先読みして最適なアクションを選択できます。

### 推奨設定

| シミュレーション回数 | 用途 | 速度 | 性能 |
|-------------------|------|------|-----|
| 10-30 | クイックテスト | 速い | 中 |
| 50-100 | バランス | 中 | 良 |
| 150-200 | 最高性能 | 遅い | 最高 |

### 使用例

```bash
# 軽量MCTS（30シミュレーション）
python gnn_any_size_emulator.py --grid-size 3 --use-mcts --num-simulations 30

# 標準MCTS（50シミュレーション）
python gnn_any_size_emulator.py --grid-size 4 --use-mcts --num-simulations 50

# 強力MCTS（100シミュレーション）
python gnn_any_size_emulator.py --grid-size 4 --use-mcts --num-simulations 100 --episodes 5
```

---

**Happy GNN Experimentation! 🎮🤖**
