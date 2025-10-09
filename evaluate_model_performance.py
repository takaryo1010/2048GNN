"""
GNN 2048 モデル性能評価スクリプト
N回プレイして統計情報を収集・分析します
"""
import numpy as np
import os
import sys
import time
from collections import Counter

# LightZeroのパスを追加
sys.path.append('./LightZero')

from lzero.entry import eval_muzero
from zoo.game_2048.config.stochastic_muzero_2048_gnn_config import main_config, create_config

def evaluate_model(model_path, num_episodes=100, seeds=None):
    """
    モデルを評価して詳細な統計を返す
    
    Args:
        model_path: モデルのパス
        num_episodes: 評価するエピソード数
        seeds: 使用するシードのリスト（Noneの場合は自動生成）
    
    Returns:
        dict: 評価結果の辞書
    """
    
    if seeds is None:
        # エピソード数に応じてシードを生成
        seeds = list(range(num_episodes))
    
    # 環境設定（動画出力なし）
    main_config.env.render_mode = None
    main_config.env.max_episode_steps = int(1e9)
    main_config.env.ignore_legal_actions = False  # 正しく終了判定
    
    # 評価設定
    create_config.env_manager.type = 'subprocess'
    main_config.env.evaluator_env_num = min(8, num_episodes)  # 並列実行
    main_config.env.n_evaluator_episode = num_episodes
    
    print("=" * 80)
    print("GNN 2048 モデル性能評価")
    print("=" * 80)
    print(f"モデル: {os.path.basename(os.path.dirname(os.path.dirname(model_path)))}")
    print(f"評価エピソード数: {num_episodes}")
    print(f"並列環境数: {main_config.env.evaluator_env_num}")
    print("=" * 80)
    
    all_scores = []
    all_max_tiles = []
    all_steps = []
    start_time = time.time()
    
    # 各シードで評価
    for i, seed in enumerate(seeds):
        print(f"\nエピソード {i+1}/{num_episodes} (seed={seed})...", end=" ", flush=True)
        
        try:
            returns_mean, returns = eval_muzero(
                [main_config, create_config],
                seed=seed,
                num_episodes_each_seed=1,
                print_seed_details=False,
                model_path=model_path
            )
            
            score = float(returns[0][0])
            all_scores.append(score)
            
            print(f"スコア: {score:.0f}")
            
        except Exception as e:
            print(f"エラー: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    
    # 統計計算
    scores_array = np.array(all_scores)
    
    results = {
        'num_episodes': len(all_scores),
        'scores': all_scores,
        'score_mean': np.mean(scores_array),
        'score_std': np.std(scores_array),
        'score_median': np.median(scores_array),
        'score_min': np.min(scores_array),
        'score_max': np.max(scores_array),
        'score_q25': np.percentile(scores_array, 25),
        'score_q75': np.percentile(scores_array, 75),
        'elapsed_time': elapsed_time,
        'avg_time_per_episode': elapsed_time / len(all_scores) if all_scores else 0,
    }
    
    return results


def print_results(results):
    """評価結果を見やすく出力"""
    print("\n" + "=" * 80)
    print("評価結果サマリー")
    print("=" * 80)
    
    print(f"\n📊 基本統計")
    print(f"  総エピソード数: {results['num_episodes']}")
    print(f"  実行時間: {results['elapsed_time']:.1f}秒")
    print(f"  平均実行時間/エピソード: {results['avg_time_per_episode']:.2f}秒")
    
    print(f"\n🎯 スコア統計")
    print(f"  平均: {results['score_mean']:.2f}")
    print(f"  標準偏差: {results['score_std']:.2f}")
    print(f"  中央値: {results['score_median']:.2f}")
    print(f"  最小値: {results['score_min']:.2f}")
    print(f"  最大値: {results['score_max']:.2f}")
    print(f"  第1四分位数 (Q1): {results['score_q25']:.2f}")
    print(f"  第3四分位数 (Q3): {results['score_q75']:.2f}")
    
    # スコアの分布を表示
    scores = results['scores']
    print(f"\n📈 スコア分布")
    bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, 10000, 20000, float('inf')]
    labels = ['0-1k', '1-2k', '2-3k', '3-4k', '4-5k', '5-6k', '6-10k', '10-20k', '20k+']
    
    hist, _ = np.histogram(scores, bins=bins)
    for label, count in zip(labels, hist):
        if count > 0:
            percentage = count / len(scores) * 100
            bar = '█' * int(percentage / 2)
            print(f"  {label:8s}: {count:3d}回 ({percentage:5.1f}%) {bar}")
    
    print("\n" + "=" * 80)


def save_results_to_file(results, filename='evaluation_results.txt'):
    """評価結果をファイルに保存"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GNN 2048 モデル性能評価結果\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"総エピソード数: {results['num_episodes']}\n")
        f.write(f"実行時間: {results['elapsed_time']:.1f}秒\n")
        f.write(f"平均実行時間/エピソード: {results['avg_time_per_episode']:.2f}秒\n\n")
        
        f.write("スコア統計:\n")
        f.write(f"  平均: {results['score_mean']:.2f}\n")
        f.write(f"  標準偏差: {results['score_std']:.2f}\n")
        f.write(f"  中央値: {results['score_median']:.2f}\n")
        f.write(f"  最小値: {results['score_min']:.2f}\n")
        f.write(f"  最大値: {results['score_max']:.2f}\n")
        f.write(f"  Q1: {results['score_q25']:.2f}\n")
        f.write(f"  Q3: {results['score_q75']:.2f}\n\n")
        
        f.write("全スコア:\n")
        for i, score in enumerate(results['scores'], 1):
            f.write(f"  エピソード {i:3d}: {score:.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"結果を保存しました: {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GNN 2048モデルの性能評価')
    parser.add_argument('--model_path', type=str, 
                        default='./LightZero/zoo/game_2048/config/data_gnn_stochastic_mz/game_2048_gnn_npct-2_ns100_upc200_rer0.0_bs512_gnn3L128D_sparse_seed0_251008_050852/ckpt/ckpt_best.pth.tar',
                        help='評価するモデルのパス')
    parser.add_argument('--num_episodes', '-n', type=int, default=100,
                        help='評価するエピソード数（デフォルト: 100）')
    parser.add_argument('--output', '-o', type=str, default='evaluation_results.txt',
                        help='結果を保存するファイル名')
    parser.add_argument('--quick', action='store_true',
                        help='クイック評価モード（10エピソード）')
    
    args = parser.parse_args()
    
    # クイックモードの場合
    if args.quick:
        args.num_episodes = 10
        print("🚀 クイック評価モード: 10エピソード")
    
    # モデルの存在確認
    if not os.path.exists(args.model_path):
        print(f"エラー: モデルファイルが見つかりません: {args.model_path}")
        sys.exit(1)
    
    # 評価実行
    results = evaluate_model(
        model_path=args.model_path,
        num_episodes=args.num_episodes
    )
    
    # 結果表示
    print_results(results)
    
    # 結果保存
    save_results_to_file(results, args.output)
