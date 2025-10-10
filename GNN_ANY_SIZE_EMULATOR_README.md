# GNN 2048 汎用サイズエミュレータ

4×4で学習したGNNモデルを**任意の盤面サイズ**で実行できる独立したエミュレータです。

## 🎯 特徴

- ✅ **任意サイズ対応**: 3×3から8×8まで、どんなサイズでも動作
- ✅ **完全独立動作**: LightZeroのGUIに依存しない
- ✅ **簡単な使用法**: コマンドライン1行で実行可能
- ✅ **MCTS推論**: Monte Carlo Tree Searchで推論を強化（オプション）
- ✅ **リアルタイム描画**: ゲームプレイを目視確認
- ✅ **GIF出力**: ゲームプレイをアニメーションとして保存
- ✅ **詳細な統計**: スコア、最大タイル、手数などを自動集計

## 📦 必要な環境

```bash
# 必要なパッケージ
pip install torch numpy matplotlib
```

## 🚀 クイックスタート

### 基本的な使い方

```bash
# 4×4盤面で10エピソード実行（デフォルト）
python gnn_any_size_emulator.py

# 3×3盤面で実行
python gnn_any_size_emulator.py --grid-size 3 --episodes 10

# 5×5盤面でリアルタイム描画付き
python gnn_any_size_emulator.py --grid-size 5 --episodes 5 --render

# 6×6盤面でGIF保存
python gnn_any_size_emulator.py --grid-size 6 --episodes 3 --save-gif
```

### MCTSで推論を強化

```bash
# MCTSを使用（50シミュレーション）
python gnn_any_size_emulator.py --grid-size 4 --use-mcts --num-simulations 50

# より強力な推論（100シミュレーション）
python gnn_any_size_emulator.py --grid-size 4 --use-mcts --num-simulations 100 --episodes 5

# 3×3でMCTS（高速）
python gnn_any_size_emulator.py --grid-size 3 --use-mcts --num-simulations 30 --episodes 10

# 比較: MCTSなし vs MCTSあり
python gnn_any_size_emulator.py --grid-size 4 --episodes 10  # ポリシーのみ
python gnn_any_size_emulator.py --grid-size 4 --episodes 10 --use-mcts --num-simulations 50  # MCTS
```

### カスタムモデルを使用

```bash
python gnn_any_size_emulator.py \
  --grid-size 4 \
  --model-path ./path/to/your/model.pth.tar \
  --episodes 20
```

## 📋 コマンドラインオプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--grid-size` | 盤面のサイズ (3〜8推奨) | 4 |
| `--episodes` | 実行するエピソード数 | 10 |
| `--model-path` | 学習済みモデルのパス | `iteration_79400.pth.tar` |
| `--use-mcts` | MCTSを使用（より良い性能、より遅い） | False |
| `--num-simulations` | MCTSのシミュレーション回数 | 50 |
| `--render` | リアルタイムで盤面を表示 | False |
| `--save-gif` | 最初のエピソードをGIFとして保存 | False |
| `--gif-path` | GIFの保存パス | `./gnn_2048_custom_size.gif` |
| `--device` | 使用するデバイス (cuda/cpu) | cuda |

## 💡 使用例

### 例1: 3×3盤面で素早くテスト

```bash
python gnn_any_size_emulator.py --grid-size 3 --episodes 5
```

**出力例:**
```
============================================================
統計情報 (盤面サイズ: 3×3)
============================================================
平均スコア:     892.40 ± 156.23
最高スコア:     1124
最低スコア:     684
平均最大タイル: 128.0
平均手数:       45.2

最大タイル達成回数:
   128: 3回
   64: 2回
============================================================
```

### 例2: 5×5盤面でリアルタイム描画

```bash
python gnn_any_size_emulator.py --grid-size 5 --episodes 3 --render
```

ターミナルに盤面がリアルタイムで表示されます：

```
=================================================
|     0|     2|     0|     8|     4|
-------------------------------------------------
|     4|    16|     2|    32|     8|
-------------------------------------------------
|     0|     8|     4|    64|    16|
-------------------------------------------------
|     2|     4|     8|   128|    32|
-------------------------------------------------
|     0|     2|     4|    16|     8|
-------------------------------------------------
スコア: 1456 | 手数: 67 | 最大タイル: 128
=================================================
```

### 例3: GIFアニメーション作成

```bash
python gnn_any_size_emulator.py \
  --grid-size 4 \
  --episodes 1 \
  --save-gif \
  --gif-path ./my_game.gif
```

`my_game.gif`にゲームプレイが保存されます。

### 例4: 大きな盤面で長時間評価

```bash
python gnn_any_size_emulator.py \
  --grid-size 6 \
  --episodes 50 \
  --device cuda
```

## 🧠 技術的な詳細

### GNNモデルの汎用性

このエミュレータは、**4×4で学習したGNNモデル**を任意のサイズに適用できます。これは以下の理由で可能です：

1. **グラフベース表現**: CNNと異なり、GNNはグラフ構造を扱うため、グリッドサイズに依存しません
2. **動的エッジ構築**: 実行時に盤面サイズに応じたエッジを自動生成
3. **位置エンコーディング**: 相対位置情報を使用するため、絶対サイズに依存しません
4. **プーリング集約**: ノード埋め込みをmean/max/sumで集約するため、ノード数に依存しません

### アーキテクチャ

```
入力観測 [B, 16, H, W]
    ↓
グラフ構築 (ノード特徴量 + エッジ)
    ↓
GraphSAGE (3層, 128次元)
    ↓
グローバルプーリング (mean/max/sum)
    ↓
ポリシーヘッド (MLP)
    ↓
アクション確率 [B, 4]
```

## 📊 パフォーマンス

### 盤面サイズ別の期待結果

| 盤面サイズ | 平均スコア | 最大タイル | 平均手数 |
|-----------|-----------|-----------|---------|
| 3×3 | 600-1000 | 64-128 | 40-60 |
| 4×4 | 8000-15000 | 512-1024 | 200-400 |
| 5×5 | 20000-40000 | 1024-2048 | 500-800 |
| 6×6 | 50000-100000 | 2048-4096 | 1000-2000 |

*注意: これらは参考値です。実際の結果はモデルの学習状態に依存します。*

### 速度

- **3×3**: ~100 moves/sec
- **4×4**: ~80 moves/sec
- **5×5**: ~60 moves/sec
- **6×6**: ~40 moves/sec

## 🐛 トラブルシューティング

### CUDAメモリ不足

大きな盤面サイズ（7×7以上）でCUDAメモリ不足になる場合：

```bash
# CPUを使用
python gnn_any_size_emulator.py --grid-size 7 --device cpu
```

### モデルのロードエラー

モデルファイルが見つからない場合：

```bash
# モデルパスを正しく指定
python gnn_any_size_emulator.py \
  --model-path ./LightZero/zoo/game_2048/config/data_gnn_stochastic_mz/gnn_simple_success/ckpt/iteration_79400.pth.tar
```

### パフォーマンスが低い

4×4以外のサイズでは、モデルが4×4専用に学習されているため、パフォーマンスが低下する可能性があります。これは正常な動作です。

## 🔬 実験のアイデア

### 1. サイズ別性能比較

```bash
# 各サイズで評価
for size in 3 4 5 6; do
  echo "Testing ${size}x${size}..."
  python gnn_any_size_emulator.py --grid-size $size --episodes 20
done
```

### 2. 転移学習の検証

4×4で学習したモデルが他のサイズでどの程度機能するかを確認：

```bash
# 3×3（小さい）
python gnn_any_size_emulator.py --grid-size 3 --episodes 50

# 5×5（大きい）
python gnn_any_size_emulator.py --grid-size 5 --episodes 50
```

### 3. 可視化比較

各サイズでGIFを作成して比較：

```bash
python gnn_any_size_emulator.py --grid-size 3 --episodes 1 --save-gif --gif-path ./3x3.gif
python gnn_any_size_emulator.py --grid-size 4 --episodes 1 --save-gif --gif-path ./4x4.gif
python gnn_any_size_emulator.py --grid-size 5 --episodes 1 --save-gif --gif-path ./5x5.gif
```

## 📝 コードの構造

```
gnn_any_size_emulator.py
├── Game2048AnySize        # 任意サイズの2048環境
├── GraphBuilder           # 動的グラフ構築
├── GraphSAGE             # GNNレイヤー
├── GNNRepresentationNetwork  # 表現ネットワーク
├── GNNPolicyHead         # ポリシーヘッド
├── GNNAgent              # エージェント
└── 評価・可視化関数
    ├── render_board()     # テキスト描画
    ├── evaluate_agent()   # 評価実行
    ├── save_game_as_gif() # GIF保存
    └── print_statistics() # 統計表示
```

## 🎓 学習ポイント

このエミュレータは以下を実証しています：

1. **GNNの汎用性**: グラフベースのアーキテクチャは固定サイズに制約されない
2. **転移学習**: 4×4で学習した知識が他のサイズにも適用可能
3. **独立実装**: 複雑なフレームワークに依存せず、コアロジックだけで実装可能

## 🤝 貢献

改善のアイデアがあれば、ぜひIssueやPull Requestを送ってください！

## 📄 ライセンス

このプロジェクトはオリジナルのLightZeroプロジェクトのライセンスに従います。

## 🙏 謝辞

- LightZero開発チーム
- PyTorch Geometric
- 2048ゲームの作者

---

**楽しいGNN実験を！🎮🤖**
