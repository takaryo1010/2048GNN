#!/bin/bash

# MCTS vs ポリシーのみ - 性能比較スクリプト

echo "========================================"
echo "GNN 2048 - MCTS vs ポリシーのみ 比較"
echo "========================================"
echo ""

MODEL_PATH="./LightZero/zoo/game_2048/config/data_gnn_stochastic_mz/gnn_simple_success/ckpt/iteration_79400.pth.tar"

# 3×3盤面での比較
echo "【テスト 1】3×3盤面 - ポリシーのみ"
echo "----------------------------------------"
python gnn_any_size_emulator.py \
    --grid-size 3 \
    --episodes 10 \
    --model-path "$MODEL_PATH" 2>&1 | grep -A 10 "統計情報"
echo ""

echo "【テスト 2】3×3盤面 - MCTS (30シミュレーション)"
echo "----------------------------------------"
python gnn_any_size_emulator.py \
    --grid-size 3 \
    --episodes 10 \
    --use-mcts \
    --num-simulations 30 \
    --model-path "$MODEL_PATH" 2>&1 | grep -A 10 "統計情報"
echo ""

# 4×4盤面での比較
echo "【テスト 3】4×4盤面 - ポリシーのみ"
echo "----------------------------------------"
python gnn_any_size_emulator.py \
    --grid-size 4 \
    --episodes 5 \
    --model-path "$MODEL_PATH" 2>&1 | grep -A 10 "統計情報"
echo ""

echo "【テスト 4】4×4盤面 - MCTS (50シミュレーション)"
echo "----------------------------------------"
python gnn_any_size_emulator.py \
    --grid-size 4 \
    --episodes 5 \
    --use-mcts \
    --num-simulations 50 \
    --model-path "$MODEL_PATH" 2>&1 | grep -A 10 "統計情報"
echo ""

echo "========================================"
echo "比較テスト完了！"
echo "========================================"
echo ""
echo "まとめ:"
echo "- MCTSは推論に時間がかかりますが、より良いスコアを出す傾向があります"
echo "- シミュレーション回数が多いほど性能は向上しますが、計算時間も増加します"
echo "- 3×3のような小さい盤面では、MCTSの効果が顕著に現れます"
