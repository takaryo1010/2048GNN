#!/bin/bash
# Quick test to verify GNN speedup in actual training

echo "=================================="
echo "GNN Speedup Verification Test"
echo "=================================="
echo ""
echo "Testing optimized GNN model with:"
echo "- Batched graph processing"
echo "- LayerNorm (no transpose)"
echo "- Sparse edge connectivity"
echo ""

cd /opendilab/2048GNN/LightZero

# Run for 30 seconds and check training speed
echo "Running training for 30 seconds..."
timeout 30 python zoo/game_2048/config/stochastic_muzero_2048_gnn_config.py 2>&1 | \
    grep -E "(train_iter|policy_loss|fps|samples/s)" | head -20

echo ""
echo "Test completed!"
echo ""
echo "Expected improvements:"
echo "  - Training FPS should be 5-10x higher than before"
echo "  - No 'for loop' delays in graph processing"
echo "  - Smooth GPU utilization"
