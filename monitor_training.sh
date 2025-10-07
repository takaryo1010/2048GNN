#!/bin/bash
# GNN Stochastic MuZero トレーニング監視スクリプト

LOG_FILE="/tmp/gnn_training_full.log"

echo "=========================================="
echo "GNN Stochastic MuZero トレーニング監視"
echo "=========================================="
echo ""

# プロセス確認
if pgrep -f "stochastic_muzero_2048_gnn_config.py" > /dev/null; then
    echo "✓ トレーニングプロセス: 実行中"
    PID=$(pgrep -f "stochastic_muzero_2048_gnn_config.py")
    echo "  PID: $PID"
else
    echo "✗ トレーニングプロセス: 停止"
fi
echo ""

# ログファイルサイズ
if [ -f "$LOG_FILE" ]; then
    SIZE=$(du -h "$LOG_FILE" | cut -f1)
    echo "ログファイルサイズ: $SIZE"
else
    echo "ログファイルが見つかりません"
    exit 1
fi
echo ""

# 最新の評価結果
echo "=========================================="
echo "最新の評価結果:"
echo "=========================================="
grep -E "EVALUATOR.*finish episode" "$LOG_FILE" | tail -5
echo ""

# トレーニングイテレーション
echo "=========================================="
echo "トレーニングイテレーション:"
echo "=========================================="
grep -E "Training Iteration.*Result" "$LOG_FILE" | tail -5
echo ""

# 損失の推移
echo "=========================================="
echo "最新の損失値:"
echo "=========================================="
grep -A 10 "Training Iteration" "$LOG_FILE" | grep -E "(policy_loss|value_loss|reward_loss|total_loss)" | tail -20
echo ""

# チェックポイント
echo "=========================================="
echo "保存されたチェックポイント:"
echo "=========================================="
if [ -d "data_gnn_stochastic_mz" ]; then
    find data_gnn_stochastic_mz -name "*.pth.tar" -type f -exec ls -lh {} \; | tail -10
fi
echo ""

# GPU使用状況
echo "=========================================="
echo "GPU使用状況:"
echo "=========================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
else
    echo "nvidia-smiが利用できません"
fi
echo ""

echo "=========================================="
echo "リアルタイムログを見るには:"
echo "  tail -f $LOG_FILE"
echo "=========================================="
