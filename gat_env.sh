# 2048GNN用のLightZero環境設定
# このファイルをDockerfileに追加するか、.bashrcにsourceしてください

export PYTHONPATH="/opendilab/2048GNN/LightZero:$PYTHONPATH"
export LIGHTZERO_GAT_HOME="/opendilab/2048GNN"

# 便利なエイリアス
alias gat-train="cd /opendilab/2048GNN && python LightZero/zoo/game_2048/config/gat_stochastic_2048_config.py"
alias gat-test="cd /opendilab/2048GNN && python test_gat_model.py"
alias gat-shell="cd /opendilab/2048GNN"

echo "✓ 2048GNN GAT環境が設定されました"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  作業ディレクトリ: $LIGHTZERO_GAT_HOME"
