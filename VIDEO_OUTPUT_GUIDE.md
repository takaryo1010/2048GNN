# GNN 2048 モデル評価・動画出力ガイド

トレーニング済みのGNNベースのStochastic MuZeroモデルを使って、2048ゲームのプレイを動画形式（MP4/GIF）で出力する方法を説明します。

## 📁 モデルの場所

トレーニング済みモデル:
```
./LightZero/zoo/game_2048/config/data_gnn_stochastic_mz/game_2048_gnn_npct-2_ns100_upc200_rer0.0_bs512_gnn3L128D_sparse_seed0_251008_050852/ckpt/ckpt_best.pth.tar
```

## 🎬 動画出力スクリプト

### 1. シンプル版（推奨）- 1エピソードをMP4出力

最も簡単な方法です。1つのエピソードを素早くMP4形式で出力します。

```bash
cd /opendilab/2048GNN
python eval_gnn_simple.py
```

**出力先:** `./video_output/2048_gnn_2048.mp4`

**実行時間:** 約1-2分

---

### 2. GIF版 - 軽量な動画形式

複数のエピソードをGIF形式で出力します（ファイルサイズが小さい）。

```bash
cd /opendilab/2048GNN
python eval_gnn_gif.py
```

**出力先:** `./gif_output/`

**設定:**
- 3つのエピソード（seed 0, 1, 2）
- GIF形式で保存

---

### 3. 詳細版 - 複数エピソードをMP4出力

複数のシードで複数のエピソードを評価し、MP4形式で出力します。

```bash
cd /opendilab/2048GNN
python eval_gnn_to_video.py
```

**出力先:** `./videos_gnn_output/`

**設定（デフォルト）:**
- シード: 0, 1, 2
- 各シード5エピソード（合計15エピソード）
- MP4形式で保存

**実行時間:** 約15-20分

---

## ⚙️ カスタマイズ方法

各スクリプトの設定部分を編集することで、出力をカスタマイズできます。

### eval_gnn_simple.py のカスタマイズ例

```python
# シンプル版のカスタマイズ
seeds = [0]  # ランダムシード
num_episodes_each_seed = 1  # エピソード数
replay_format = 'mp4'  # 'mp4' または 'gif'
replay_path = './video_output'  # 出力ディレクトリ
```

### eval_gnn_to_video.py のカスタマイズ例

```python
# 詳細版のカスタマイズ
seeds = [0, 1, 2, 3, 4]  # 5つのシード
num_episodes_each_seed = 3  # 各シード3エピソード（合計15エピソード）
replay_format = 'mp4'  # または 'gif'
replay_path = './my_videos'  # カスタム出力パス
```

### eval_gnn_gif.py のカスタマイズ例

```python
# GIF版のカスタマイズ
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10エピソード
replay_format = 'gif'
replay_path = './gif_collection'
```

---

## 📊 出力例

スクリプト実行時の出力例:

```
============================================================
GNN 2048 - MP4動画出力 (シンプル版)
============================================================
モデル: game_2048_gnn_npct-2_ns100_upc200_rer0.0_bs512_gnn3L128D_sparse_seed0_251008_050852
出力先: ./video_output
============================================================

[モデル読み込み中...]
[評価実行中...]

============================================================
完了!
============================================================
エピソードの報酬: 3312.00
MP4動画保存先: ./video_output
============================================================
```

---

## 📋 出力ファイルの確認

```bash
# MP4ファイルの確認
ls -lh video_output/

# GIFファイルの確認
ls -lh gif_output/

# ファイルサイズの確認
du -sh video_output/ gif_output/
```

---

## 🎮 モデルの性能

このGNNモデルは以下の性能を持っています:

- **平均報酬:** 約3000-3500
- **到達タイル:** 2048タイル以上を安定して達成
- **学習アルゴリズム:** Stochastic MuZero with GNN
- **ネットワーク:** GraphSAGE (3層, 128次元)

---

## 🔧 トラブルシューティング

### エラー: モデルファイルが見つからない

```bash
# チェックポイントの存在確認
ls -la LightZero/zoo/game_2048/config/data_gnn_stochastic_mz/*/ckpt/
```

### エラー: メモリ不足

複数エピソードを出力する場合は、バッチサイズを減らしてください:

```python
# eval_gnn_to_video.py で
num_episodes_each_seed = 1  # 3から1に減らす
```

### 動画が再生できない

MP4ファイルが再生できない場合、GIF形式を試してください:

```python
replay_format = 'gif'
```

---

## 📝 技術仕様

### モデル詳細
- **アーキテクチャ:** GNN-based Stochastic MuZero
- **GNNタイプ:** GraphSAGE
- **レイヤー数:** 3
- **隠れ層次元:** 128
- **スパース最適化:** 有効

### 学習設定
- **シミュレーション数:** 100
- **アンロール数:** 200
- **バッチサイズ:** 512
- **リプレイ比率:** 0.0
- **学習率:** 0.003

### 評価設定
- **MCTS温度:** 1.0
- **最大ステップ数:** 無制限（ゲームが終了するまで）
- **評価環境数:** 1（動画出力のため）

---

## 🚀 次のステップ

1. **シンプル版から開始:** まず `eval_gnn_simple.py` を実行して1つの動画を確認
2. **GIF版を試す:** 複数のエピソードを軽量なGIF形式で確認
3. **詳細版で大量生成:** 満足したら複数エピソードをMP4形式で生成

---

## 📚 関連ドキュメント

- [GNN実装成功レポート](GNN_IMPLEMENTATION_SUCCESS.md)
- [GNN最適化レポート](GNN_OPTIMIZATION_COMPLETE.md)
- [速度比較レポート](SPEED_COMPARISON_REPORT.md)

---

## ⚡ 高速化のヒント

動画生成を高速化するには:

1. **エピソード数を減らす**
   ```python
   num_episodes_each_seed = 1
   ```

2. **GIF形式を使用**（エンコードが速い）
   ```python
   replay_format = 'gif'
   ```

3. **並列実行は非推奨**（動画出力は逐次実行が必要）

---

**作成日:** 2025年10月8日  
**モデルバージョン:** 251008_050852
