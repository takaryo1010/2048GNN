"""
GNN 2048 モデル性能評価スクリプト（シンプル版）
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


def estimate_max_tile(score):
    """スコアから最大タイルを推定"""
    if score < 500:
        return 128
    elif score < 1000:
        return 256
    elif score < 2000:
        return 512
    elif score < 3500:
        return 1024
    elif score < 6000:
        return 2048
    elif score < 10000:
        return 4096
    elif score < 20000:
        return 8192
    elif score < 40000:
        return 16384
    else:
        return 32768


def evaluate_model(model_path, num_episodes=10, save_video=True, video_dir='./video_output'):
    """モデルを評価して詳細な統計を返す"""
    
    # 動画出力ディレクトリの作成
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
    
    # 逐次実行設定
    create_config.env_manager.type = 'base'
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    main_config.env.max_episode_steps = int(1e9)
    main_config.env.ignore_legal_actions = False  # 正しく終了判定
    
    print("=" * 80)
    print("🎮 GNN 2048 モデル性能評価")
    print("=" * 80)
    print(f"📁 モデル: {os.path.basename(os.path.dirname(os.path.dirname(model_path)))}")
    print(f"📊 評価エピソード数: {num_episodes}")
    print(f"⚙️  実行モード: 逐次実行")
    print(f"🎬 動画出力: {'有効' if save_video else '無効'}")
    if save_video:
        print(f"📂 動画保存先: {video_dir}")
    print("=" * 80)
    
    all_scores = []
    all_max_tiles = []
    all_video_paths = []
    start_time = time.time()
    
    # 各エピソードを評価
    for i in range(num_episodes):
        seed = i
        
        # 各エピソードで動画出力を設定
        if save_video:
            main_config.env.render_mode = 'rgb_array'
            main_config.env.save_replay = True
            main_config.env.replay_path = os.path.join(video_dir, f'episode_{i+1:03d}_seed{seed}.gif')
        else:
            main_config.env.render_mode = None
            main_config.env.save_replay = False
        
        print(f"\rエピソード {i+1}/{num_episodes}...", end="", flush=True)
        
        try:
            returns_mean, returns = eval_muzero(
                [main_config, create_config],
                seed=seed,
                num_episodes_each_seed=1,
                print_seed_details=False,
                model_path=model_path
            )
            
            score = float(returns[0][0])
            max_tile = estimate_max_tile(score)
            
            all_scores.append(score)
            all_max_tiles.append(max_tile)
            
            # 動画パスを記録
            if save_video:
                video_path = main_config.env.replay_path
                all_video_paths.append(video_path)
            
            # 10エピソードごとに進捗表示
            if (i + 1) % 10 == 0:
                avg_score = np.mean(all_scores)
                print(f"\rエピソード {i+1}/{num_episodes} 完了 (平均スコア: {avg_score:.0f})        ")
            
        except Exception as e:
            print(f"\n❌ エラー (エピソード {i+1}): {e}")
            continue
    
    print()  # 改行
    elapsed_time = time.time() - start_time
    
    # 統計計算
    scores_array = np.array(all_scores)
    max_tiles_array = np.array(all_max_tiles)
    
    results = {
        'num_episodes': len(all_scores),
        'elapsed_time': elapsed_time,
        'avg_time_per_episode': elapsed_time / len(all_scores) if all_scores else 0,
        
        # スコア統計
        'scores': all_scores,
        'score_mean': np.mean(scores_array),
        'score_std': np.std(scores_array),
        'score_median': np.median(scores_array),
        'score_min': np.min(scores_array),
        'score_max': np.max(scores_array),
        'score_q25': np.percentile(scores_array, 25),
        'score_q75': np.percentile(scores_array, 75),
        
        # 最大タイル統計
        'max_tiles': all_max_tiles,
        'max_tile_counter': Counter(all_max_tiles),
        'max_tile_mode': Counter(all_max_tiles).most_common(1)[0][0] if all_max_tiles else 0,
        
        # 動画パス
        'video_paths': all_video_paths if save_video else [],
        'save_video': save_video,
        'video_dir': video_dir if save_video else None,
    }
    
    return results


def print_results(results):
    """評価結果を見やすく表示"""
    print("\n" + "=" * 80)
    print("📈 評価結果サマリー")
    print("=" * 80)
    
    print(f"\n⏱️  実行情報")
    print(f"  総エピソード数: {results['num_episodes']}")
    print(f"  実行時間: {results['elapsed_time']:.1f}秒 ({results['elapsed_time']/60:.1f}分)")
    print(f"  平均実行時間/エピソード: {results['avg_time_per_episode']:.2f}秒")
    
    print(f"\n🎯 スコア統計")
    print(f"  平均: {results['score_mean']:.2f} ± {results['score_std']:.2f}")
    print(f"  中央値: {results['score_median']:.2f}")
    print(f"  範囲: [{results['score_min']:.0f}, {results['score_max']:.0f}]")
    print(f"  四分位範囲: [{results['score_q25']:.0f}, {results['score_q75']:.0f}]")
    
    print(f"\n🏆 最大タイル達成状況")
    tile_counter = results['max_tile_counter']
    total = results['num_episodes']
    
    # タイルを大きい順にソート
    sorted_tiles = sorted(tile_counter.keys(), reverse=True)
    for tile in sorted_tiles:
        count = tile_counter[tile]
        percentage = count / total * 100
        bar = '█' * min(int(percentage / 2), 50)
        print(f"  {tile:5d}: {count:3d}回 ({percentage:5.1f}%) {bar}")
    
    # 累積達成率
    print(f"\n📊 累積達成率")
    cumulative = 0
    for tile in sorted_tiles:
        cumulative += tile_counter[tile]
        percentage = cumulative / total * 100
        print(f"  {tile}以上: {cumulative:3d}回 ({percentage:5.1f}%)")
    
    # スコア分布
    print(f"\n💯 スコア分布")
    scores = results['scores']
    bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 15000, float('inf')]
    labels = ['<1k', '1-2k', '2-3k', '3-4k', '4-5k', '5-6k', '6-8k', '8-10k', '10-15k', '15k+']
    
    hist, _ = np.histogram(scores, bins=bins)
    for label, count in zip(labels, hist):
        if count > 0:
            percentage = count / len(scores) * 100
            bar = '█' * min(int(percentage / 2), 50)
            print(f"  {label:8s}: {count:3d}回 ({percentage:5.1f}%) {bar}")
    
    # トップ10エピソード
    if results['num_episodes'] >= 10:
        print(f"\n🥇 トップ10エピソード")
        scores_with_idx = [(i, s, t) for i, (s, t) in enumerate(zip(results['scores'], results['max_tiles']))]
        top10 = sorted(scores_with_idx, key=lambda x: x[1], reverse=True)[:10]
        
        for rank, (idx, score, max_tile) in enumerate(top10, 1):
            print(f"  {rank:2d}. エピソード {idx+1:3d}: スコア {score:8.0f}, 最大タイル {max_tile}")
    
    # 動画情報
    if results.get('save_video', False):
        print(f"\n🎬 動画出力")
        print(f"  保存先: {results['video_dir']}")
        print(f"  動画数: {len(results['video_paths'])}個")
        if results['video_paths']:
            print(f"  例: {os.path.basename(results['video_paths'][0])}")
    
    print("\n" + "=" * 80)


def save_results(results, txt_file='evaluation_results.txt'):
    """評価結果をテキストファイルに保存"""
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GNN 2048 モデル性能評価結果\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"総エピソード数: {results['num_episodes']}\n")
        f.write(f"実行時間: {results['elapsed_time']:.1f}秒 ({results['elapsed_time']/60:.1f}分)\n\n")
        
        f.write("スコア統計:\n")
        f.write(f"  平均: {results['score_mean']:.2f}\n")
        f.write(f"  標準偏差: {results['score_std']:.2f}\n")
        f.write(f"  中央値: {results['score_median']:.2f}\n")
        f.write(f"  最小値: {results['score_min']:.2f}\n")
        f.write(f"  最大値: {results['score_max']:.2f}\n")
        f.write(f"  Q1: {results['score_q25']:.2f}\n")
        f.write(f"  Q3: {results['score_q75']:.2f}\n\n")
        
        f.write("最大タイル分布:\n")
        for tile, count in sorted(results['max_tile_counter'].items(), reverse=True):
            percentage = count / results['num_episodes'] * 100
            f.write(f"  {tile:5d}: {count:3d}回 ({percentage:5.1f}%)\n")
        
        f.write("\n全スコア:\n")
        for i, (score, tile) in enumerate(zip(results['scores'], results['max_tiles']), 1):
            video_info = ""
            if results.get('save_video', False) and i-1 < len(results['video_paths']):
                video_info = f", 動画: {os.path.basename(results['video_paths'][i-1])}"
            f.write(f"  エピソード {i:3d}: スコア {score:8.2f}, 最大タイル {tile}{video_info}\n")
        
        if results.get('save_video', False):
            f.write(f"\n動画出力:\n")
            f.write(f"  保存先: {results['video_dir']}\n")
            f.write(f"  動画数: {len(results['video_paths'])}個\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✅ 結果を保存しました: {txt_file}")


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
    parser.add_argument('--no-video', action='store_true',
                        help='動画出力を無効化')
    parser.add_argument('--video-dir', type=str, default='./video_output',
                        help='動画の保存先ディレクトリ')
    
    args = parser.parse_args()
    
    # クイックモード
    if args.quick:
        args.num_episodes = 10
        print("🚀 クイック評価モード: 10エピソード\n")
    
    # モデル存在確認
    if not os.path.exists(args.model_path):
        print(f"❌ エラー: モデルファイルが見つかりません: {args.model_path}")
        sys.exit(1)
    
    # 評価実行
    results = evaluate_model(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        save_video=not args.no_video,
        video_dir=args.video_dir
    )
    
    # 結果表示
    print_results(results)
    
    # 結果保存
    save_results(results, args.output)
    
    print(f"\n✨ 評価完了!")
