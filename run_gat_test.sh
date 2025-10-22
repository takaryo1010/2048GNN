#!/bin/bash
# GAT Training Test Script
# Tests GAT-based Stochastic MuZero model with minimal configuration

echo "=========================================="
echo "GAT Model Quick Training Test"
echo "=========================================="
echo ""

cd /opendilab/2048GNN
python quick_gat_training_test.py

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
