#!/bin/bash
# GAT StochasticMuZero実行スクリプト

cd /opendilab/2048GNN
export PYTHONPATH="/opendilab/2048GNN/LightZero:$PYTHONPATH"

echo "🚀 GAT StochasticMuZero 実行開始..."
echo "📂 作業ディレクトリ: $(pwd)"
echo "🐍 PYTHONPATH: 2048GNN/LightZero を優先設定"
echo ""

python LightZero/zoo/game_2048/config/gat_stochastic_2048_config.py "$@"
