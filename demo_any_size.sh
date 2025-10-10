#!/bin/bash

# GNN 2048 汎用サイズエミュレータ - デモスクリプト
# 異なる盤面サイズでGNNモデルをテストします

echo "========================================"
echo "GNN 2048 汎用サイズエミュレータ - デモ"
echo "========================================"
echo ""

# モデルパス
MODEL_PATH="./LightZero/zoo/game_2048/config/data_gnn_stochastic_mz/gnn_simple_success/ckpt/iteration_79400.pth.tar"

# 3×3盤面でテスト
echo "【テスト 1/4】3×3盤面で5エピソード実行"
echo "----------------------------------------"
python gnn_any_size_emulator.py \
    --grid-size 3 \
    --episodes 5 \
    --model-path "$MODEL_PATH"
echo ""

# 4×4盤面でテスト（元のサイズ）
echo "【テスト 2/4】4×4盤面で5エピソード実行（元のサイズ）"
echo "----------------------------------------"
python gnn_any_size_emulator.py \
    --grid-size 4 \
    --episodes 5 \
    --model-path "$MODEL_PATH"
echo ""

# 5×5盤面でテスト
echo "【テスト 3/4】5×5盤面で3エピソード実行"
echo "----------------------------------------"
python gnn_any_size_emulator.py \
    --grid-size 5 \
    --episodes 3 \
    --model-path "$MODEL_PATH"
echo ""

# 6×6盤面でテスト
echo "【テスト 4/4】6×6盤面で2エピソード実行"
echo "----------------------------------------"
python gnn_any_size_emulator.py \
    --grid-size 6 \
    --episodes 2 \
    --model-path "$MODEL_PATH"
echo ""

echo "========================================"
echo "全テスト完了！"
echo "========================================"
echo ""
echo "詳細な使い方は GNN_ANY_SIZE_EMULATOR_README.md を参照してください"
