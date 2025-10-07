#!/bin/bash
# GNN Stochastic MuZero 3x3 トレーニング監視スクリプト

LOG_FILE="/tmp/gnn_3x3_training.log"

echo "=========================================="
echo "GNN Stochastic MuZero トレーニング監視 (3x3)"
echo "=========================================="
echo ""

# プロセス確認
if pgrep -f "stochastic_muzero_2048_gnn_3x3_config.py" > /dev/null; then
    echo "✓ トレーニングプロセス: 実行中"
    PID=$(pgrep -f "stochastic_muzero_2048_gnn_3x3_config.py" | head -1)
    echo "  メインPID: $PID"
    ELAPSED=$(ps -p $PID -o etime= | tr -d ' ')
    echo "  実行時間: $ELAPSED"
else
    echo "✗ トレーニングプロセス: 停止"
fi
echo ""

# ログファイルサイズ
if [ -f "$LOG_FILE" ]; then
    SIZE=$(du -h "$LOG_FILE" | cut -f1)
    LINES=$(wc -l < "$LOG_FILE")
    echo "ログファイル: $SIZE ($LINES 行)"
else
    echo "ログファイルが見つかりません"
    exit 1
fi
echo ""

# 最新の評価結果
echo "=========================================="
echo "最新の評価結果 (最後の5回):"
echo "=========================================="
grep -E "EVALUATOR.*finish episode.*final reward" "$LOG_FILE" | tail -5
echo ""

# トレーニングイテレーション
echo "=========================================="
echo "トレーニングイテレーション:"
echo "=========================================="
ITERATIONS=$(grep -c "Training Iteration.*Result" "$LOG_FILE")
echo "完了したイテレーション数: $ITERATIONS"
grep -E "Training Iteration.*Result" "$LOG_FILE" | tail -3
echo ""

# 最新の損失値
echo "=========================================="
echo "最新の損失値 (最後のイテレーション):"
echo "=========================================="
grep -A 20 "Training Iteration.*Result" "$LOG_FILE" | tail -30 | grep -E "(policy_loss_avg|value_loss_avg|reward_loss_avg|total_loss_avg|afterstate_policy_loss_avg)" | tail -5
echo ""

# 最新の予測値
echo "=========================================="
echo "最新の予測値:"
echo "=========================================="
grep -A 20 "Training Iteration.*Result" "$LOG_FILE" | tail -30 | grep -E "(predicted_rewards_avg|predicted_values_avg|target)" | tail -4
echo ""

# チェックポイント
echo "=========================================="
echo "保存されたチェックポイント:"
echo "=========================================="
if [ -d "data_gnn_stochastic_mz_3x3" ]; then
    CKPT_COUNT=$(find data_gnn_stochastic_mz_3x3 -name "*.pth.tar" -type f | wc -l)
    echo "チェックポイント数: $CKPT_COUNT"
    find data_gnn_stochastic_mz_3x3 -name "*.pth.tar" -type f -exec ls -lh {} \; | tail -5
else
    echo "チェックポイントディレクトリが見つかりません"
fi
echo ""

# GPU使用状況
echo "=========================================="
echo "GPU使用状況:"
echo "=========================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
else
    echo "nvidia-smiが利用できません"
fi
echo ""

# エラーチェック
echo "=========================================="
echo "エラーチェック:"
echo "=========================================="
ERROR_COUNT=$(grep -c -E "(Error|Exception|Traceback)" "$LOG_FILE" 2>/dev/null || echo "0")
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "⚠ エラーが検出されました: $ERROR_COUNT 件"
    echo "最新のエラー:"
    grep -E "(Error|Exception)" "$LOG_FILE" | tail -3
else
    echo "✓ エラーなし"
fi
echo ""

echo "=========================================="
echo "リアルタイムログを見るには:"
echo "  tail -f $LOG_FILE"
echo ""
echo "トレーニングを停止するには:"
echo "  pkill -f stochastic_muzero_2048_gnn_3x3_config.py"
echo "=========================================="
