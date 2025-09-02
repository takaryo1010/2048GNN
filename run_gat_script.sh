#!/bin/bash
# GAT関連スクリプト実行の汎用ラッパー

cd /opendilab/2048GNN
export PYTHONPATH="/opendilab/2048GNN/LightZero:$PYTHONPATH"

if [ -z "$1" ]; then
    echo "使用方法: $0 <スクリプトファイル> [引数...]"
    echo "例: $0 LightZero/zoo/game_2048/config/gat_stochastic_2048_config.py"
    echo "例: $0 test_gat_model.py"
    exit 1
fi

echo "🚀 GAT環境でスクリプト実行: $1"
echo "📂 作業ディレクトリ: $(pwd)"
echo "🐍 PYTHONPATH: 2048GNN/LightZero を優先設定"
echo ""

python "$@"
