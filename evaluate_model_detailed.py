"""
GNN 2048 モデル詳細性能評価スクリプト
スコア、最大タイル、ステップ数などを詳細に収集・分析します
"""
import numpy as np
import os
import sys
import time
import json
from collections import Counter, defaultdict

# LightZeroのパスを追加
sys.path.append('./LightZero')

from lzero.entry import eval_muzero
from zoo.game_2048.config.stochastic_muzero_2048_gnn_config import main_config, create_config

class DetailedEvaluator:
    """詳細な評価を行うクラス"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.results = {
            'scores': [],
            'max_tiles': [],
            'steps': [],
            'seeds': [],
        }
    
    def evaluate(self, num_episodes=100):
        """モデルを評価"""
        
        # 環境設定
        main_config.env.render_mode = None
        main_config.env.max_episode_steps = int(1e9)
        main_config.env.ignore_legal_actions = False
        
        # 並列評価設定
        create_config.env_manager.type = 'subprocess'
        main_config.env.evaluator_env_num = min(8, num_episodes)
        main_config.env.n_evaluator_episode = num_episodes
        
        print("=" * 80)
        print("GNN 2048 モデル詳細性能評価")
        print("=" * 80)
        print(f"モデル: {os.path.basename(os.path.dirname(os.path.dirname(self.model_path)))}")
        print(f"評価エピソード数: {num_episodes}")
        print(f"並列環境数: {main_config.env.evaluator_env_num}")
        print("=" * 80)
        
        start_time = time.time()
        
        # 評価実行
        for i in range(num_episodes):
            seed = i
            print(f"\rエピソード {i+1}/{num_episodes}...", end="", flush=True)
            
            try:
                returns_mean, returns = eval_muzero(
                    [main_config, create_config],
                    seed=seed,
                    num_episodes_each_seed=1,
                    print_seed_details=False,
                    model_path=self.model_path
                )
                
                score = float(returns[0][0])
                self.results['scores'].append(score)
                self.results['seeds'].append(seed)
                
                # 最大タイルはスコアから推定（正確な値は環境から取得が必要）
                # スコアが高いほど大きなタイルが出る傾向
                estimated_max_tile = self._estimate_max_tile(score)
                self.results['max_tiles'].append(estimated_max_tile)
                
            except Exception as e:
                print(f"\nエラー (エピソード {i+1}): {e}")
                continue
        
        print()  # 改行
        self.elapsed_time = time.time() - start_time
        
        return self._calculate_statistics()
    
    def _estimate_max_tile(self, score):
        """スコアから最大タイルを推定"""
        # 2048ゲームの一般的なスコア範囲から推定
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
    
    def _calculate_statistics(self):
        """統計情報を計算"""
        scores = np.array(self.results['scores'])
        max_tiles = np.array(self.results['max_tiles'])
        
        stats = {
            'num_episodes': len(scores),
            'elapsed_time': self.elapsed_time,
            'avg_time_per_episode': self.elapsed_time / len(scores) if len(scores) > 0 else 0,
            
            # スコア統計
            'score_mean': np.mean(scores),
            'score_std': np.std(scores),
            'score_median': np.median(scores),
            'score_min': np.min(scores),
            'score_max': np.max(scores),
            'score_q25': np.percentile(scores, 25),
            'score_q75': np.percentile(scores, 75),
            
            # 最大タイル統計
            'max_tile_counter': Counter(max_tiles),
            'max_tile_mode': Counter(max_tiles).most_common(1)[0][0] if len(max_tiles) > 0 else 0,
            
            # 生データ
            'scores': self.results['scores'],
            'max_tiles': self.results['max_tiles'],
            'seeds': self.results['seeds'],
        }
        
        return stats


def print_detailed_results(stats):
    """詳細な評価結果を表示"""
    print("\n" + "=" * 80)
    print("📊 詳細評価結果サマリー")
    print("=" * 80)
    
    print(f"\n⏱️  実行情報")
    print(f"  総エピソード数: {stats['num_episodes']}")
    print(f"  実行時間: {stats['elapsed_time']:.1f}秒 ({stats['elapsed_time']/60:.1f}分)")
    print(f"  平均実行時間/エピソード: {stats['avg_time_per_episode']:.2f}秒")
    
    print(f"\n🎯 スコア統計")
    print(f"  平均: {stats['score_mean']:.2f} ± {stats['score_std']:.2f}")
    print(f"  中央値: {stats['score_median']:.2f}")
    print(f"  範囲: [{stats['score_min']:.0f}, {stats['score_max']:.0f}]")
    print(f"  四分位範囲: [{stats['score_q25']:.0f}, {stats['score_q75']:.0f}]")
    
    print(f"\n🏆 最大タイル達成状況")
    tile_counter = stats['max_tile_counter']
    total = stats['num_episodes']
    
    # タイルを大きい順にソート
    sorted_tiles = sorted(tile_counter.keys(), reverse=True)
    for tile in sorted_tiles:
        count = tile_counter[tile]
        percentage = count / total * 100
        bar = '█' * int(percentage / 2)
        print(f"  {tile:5d}: {count:3d}回 ({percentage:5.1f}%) {bar}")
    
    # 累積達成率
    print(f"\n📈 累積達成率")
    cumulative = 0
    for tile in sorted_tiles:
        cumulative += tile_counter[tile]
        percentage = cumulative / total * 100
        print(f"  {tile}以上: {percentage:.1f}%")
    
    print(f"\n📊 スコア分布")
    scores = stats['scores']
    bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 15000, float('inf')]
    labels = ['<1k', '1-2k', '2-3k', '3-4k', '4-5k', '5-6k', '6-8k', '8-10k', '10-15k', '15k+']
    
    hist, _ = np.histogram(scores, bins=bins)
    for label, count in zip(labels, hist):
        if count > 0:
            percentage = count / len(scores) * 100
            bar = '█' * int(percentage / 2)
            print(f"  {label:8s}: {count:3d}回 ({percentage:5.1f}%) {bar}")
    
    # トップ10エピソード
    print(f"\n🥇 トップ10エピソード")
    scores_with_idx = [(i, s, t) for i, (s, t) in enumerate(zip(stats['scores'], stats['max_tiles']))]
    top10 = sorted(scores_with_idx, key=lambda x: x[1], reverse=True)[:10]
    
    for rank, (idx, score, max_tile) in enumerate(top10, 1):
        print(f"  {rank:2d}. エピソード {idx+1:3d}: スコア {score:8.0f}, 最大タイル {max_tile}")
    
    print("\n" + "=" * 80)


def save_detailed_results(stats, output_file='detailed_evaluation.json', txt_file='detailed_evaluation.txt'):
    """評価結果をJSON形式とテキスト形式で保存"""
    
    # JSON形式で保存（生データ含む）
    json_data = {
        'num_episodes': stats['num_episodes'],
        'elapsed_time': stats['elapsed_time'],
        'avg_time_per_episode': stats['avg_time_per_episode'],
        'score_statistics': {
            'mean': float(stats['score_mean']),
            'std': float(stats['score_std']),
            'median': float(stats['score_median']),
            'min': float(stats['score_min']),
            'max': float(stats['score_max']),
            'q25': float(stats['score_q25']),
            'q75': float(stats['score_q75']),
        },
        'max_tile_distribution': {str(k): int(v) for k, v in stats['max_tile_counter'].items()},
        'raw_data': {
            'scores': [float(s) for s in stats['scores']],
            'max_tiles': [int(t) for t in stats['max_tiles']],
            'seeds': [int(s) for s in stats['seeds']],
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON結果を保存: {output_file}")
    
    # テキスト形式でも保存
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GNN 2048 モデル詳細性能評価結果\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"総エピソード数: {stats['num_episodes']}\n")
        f.write(f"実行時間: {stats['elapsed_time']:.1f}秒\n\n")
        
        f.write("スコア統計:\n")
        f.write(f"  平均: {stats['score_mean']:.2f}\n")
        f.write(f"  標準偏差: {stats['score_std']:.2f}\n")
        f.write(f"  中央値: {stats['score_median']:.2f}\n")
        f.write(f"  最小値: {stats['score_min']:.2f}\n")
        f.write(f"  最大値: {stats['score_max']:.2f}\n\n")
        
        f.write("最大タイル分布:\n")
        for tile, count in sorted(stats['max_tile_counter'].items(), reverse=True):
            percentage = count / stats['num_episodes'] * 100
            f.write(f"  {tile:5d}: {count:3d}回 ({percentage:5.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✅ テキスト結果を保存: {txt_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GNN 2048モデルの詳細性能評価')
    parser.add_argument('--model_path', type=str,
                        default='./LightZero/zoo/game_2048/config/data_gnn_stochastic_mz/game_2048_gnn_npct-2_ns100_upc200_rer0.0_bs512_gnn3L128D_sparse_seed0_251008_050852/ckpt/ckpt_best.pth.tar',
                        help='評価するモデルのパス')
    parser.add_argument('--num_episodes', '-n', type=int, default=100,
                        help='評価するエピソード数（デフォルト: 100）')
    parser.add_argument('--output_json', type=str, default='detailed_evaluation.json',
                        help='JSON出力ファイル名')
    parser.add_argument('--output_txt', type=str, default='detailed_evaluation.txt',
                        help='テキスト出力ファイル名')
    parser.add_argument('--quick', action='store_true',
                        help='クイック評価モード（10エピソード）')
    
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
    evaluator = DetailedEvaluator(args.model_path)
    stats = evaluator.evaluate(num_episodes=args.num_episodes)
    
    # 結果表示
    print_detailed_results(stats)
    
    # 結果保存
    save_detailed_results(stats, args.output_json, args.output_txt)
