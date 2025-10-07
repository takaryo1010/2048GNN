#!/bin/bash
# Test training with optimized GNN model and monitor performance

echo "Starting GNN Training Speed Test"
echo "=================================="
echo ""

# Clean up any old processes
pkill -f stochastic_muzero_2048_gnn_config.py 2>/dev/null
sleep 2

# Start training in background
cd /opendilab/2048GNN/LightZero
nohup python zoo/game_2048/config/stochastic_muzero_2048_gnn_config.py > /tmp/gnn_speed_test.log 2>&1 &
PID=$!

echo "Training started (PID: $PID)"
echo "Waiting for training to initialize..."
sleep 20

echo ""
echo "Checking training progress..."
echo "=================================="

# Check if process is still running
if ps -p $PID > /dev/null; then
    echo "✓ Training process is running"
else
    echo "✗ Training process died"
    echo ""
    echo "Last 50 lines of log:"
    tail -50 /tmp/gnn_speed_test.log
    exit 1
fi

echo ""
echo "Recent log output:"
echo "--------------------------------"
tail -30 /tmp/gnn_speed_test.log | grep -v "FutureWarning\|Please install"

echo ""
echo "Waiting for first training iteration..."
sleep 40

echo ""
echo "Training statistics:"
echo "=================================="
grep -E "train_iter|policy_loss|value_loss|train_time" /tmp/gnn_speed_test.log | head -10

echo ""
echo "Collecting speed metrics..."
sleep 20

echo ""
echo "Final speed check:"
echo "=================================="
grep -E "train_iter|policy_loss|value_loss|reward_loss|train_time|collect_time" /tmp/gnn_speed_test.log | tail -15

# Stop training
echo ""
echo "Stopping training process..."
kill $PID 2>/dev/null
sleep 2

echo ""
echo "=================================="
echo "Test Complete!"
echo "=================================="
echo ""
echo "Full log available at: /tmp/gnn_speed_test.log"
echo ""
echo "To check training metrics:"
echo "  grep -E 'train_iter|loss|time' /tmp/gnn_speed_test.log"
