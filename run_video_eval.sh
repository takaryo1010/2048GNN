#!/bin/bash
# GNN 2048 モデル - 動画出力クイックスタート

echo "=========================================="
echo "GNN 2048 動画出力 - クイックスタート"
echo "=========================================="
echo ""

# ディレクトリ移動
cd /opendilab/2048GNN

echo "選択してください:"
echo "1) シンプル版 - 1エピソードをMP4出力（推奨）"
echo "2) GIF版 - 3エピソードをGIF出力"
echo "3) 詳細版 - 15エピソードをMP4出力（時間がかかります）"
echo ""
read -p "選択 (1-3): " choice

case $choice in
    1)
        echo ""
        echo "シンプル版を実行します..."
        python eval_gnn_simple.py
        echo ""
        echo "✓ 完了! 動画を確認: ./video_output/2048_gnn_2048.mp4"
        ;;
    2)
        echo ""
        echo "GIF版を実行します..."
        python eval_gnn_gif.py
        echo ""
        echo "✓ 完了! 動画を確認: ./gif_output/"
        ;;
    3)
        echo ""
        echo "詳細版を実行します（15-20分かかります）..."
        python eval_gnn_to_video.py
        echo ""
        echo "✓ 完了! 動画を確認: ./videos_gnn_output/"
        ;;
    *)
        echo "無効な選択です"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
