"""
GAT vs CNN Performance Comparison Script
Compares performance at the same number of steps
"""
import re

def parse_full_log(log_path):
    """ログファイルから全エピソードの統計情報を抽出"""
    with open(log_path, 'r') as f:
        content = f.read()
    
    episodes = re.findall(r'collect end:(.*?)(?=collect end:|$)', content, re.DOTALL)
    
    data = []
    for episode in episodes:
        ep_data = {}
        
        # 各統計値を抽出
        patterns = {
            'episode_count': r'episode_count: (\d+)',
            'envstep_count': r'envstep_count: (\d+)',
            'total_envstep_count': r'total_envstep_count: (\d+)',
            'total_episode_count': r'total_episode_count: (\d+)',
            'reward_mean': r'reward_mean: ([0-9.]+)',
            'reward_std': r'reward_std: ([0-9.]+)',
            'reward_max': r'reward_max: ([0-9.]+)',
            'reward_min': r'reward_min: ([0-9.]+)',
            'avg_envstep_per_episode': r'avg_envstep_per_episode: ([0-9.]+)',
            'avg_envstep_per_sec': r'avg_envstep_per_sec: ([0-9.]+)',
            'avg_episode_per_sec': r'avg_episode_per_sec: ([0-9.]+)',
            'collect_time': r'collect_time: ([0-9.]+)',
            'total_duration': r'total_duration: ([0-9.]+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, episode)
            if match:
                ep_data[key] = float(match.group(1))
        
        if ep_data:
            data.append(ep_data)
    
    return data


def filter_by_steps(data, max_steps):
    """指定ステップ数以下のデータのみを返す"""
    return [d for d in data if d.get('total_envstep_count', 0) <= max_steps]


def compute_stats(data, last_n=None):
    """統計値を計算"""
    if last_n:
        data = data[-last_n:]
    
    if not data:
        return {}
    
    stats = {}
    keys = ['reward_mean', 'reward_max', 'reward_min', 'avg_envstep_per_sec', 
            'avg_episode_per_sec', 'collect_time', 'avg_envstep_per_episode']
    
    for key in keys:
        values = [d[key] for d in data if key in d]
        if values:
            stats[f'{key}_avg'] = sum(values) / len(values)
            stats[f'{key}_std'] = (sum((x - stats[f'{key}_avg'])**2 for x in values) / len(values))**0.5
            stats[f'{key}_min'] = min(values)
            stats[f'{key}_max'] = max(values)
    
    # 最終的な累積値
    if data:
        stats['total_envstep_count'] = data[-1].get('total_envstep_count', 0)
        stats['total_episode_count'] = data[-1].get('total_episode_count', 0)
        stats['total_duration'] = data[-1].get('total_duration', 0)
    
    return stats


def main():
    # ファイルパス
    gat_log = "LightZero/data_gat_stochastic_mz/game_2048_gat_npct-2_ns100_upc200_rer0.0_bs512_gat3L128D_h4_sparse_seed0_251022_034716/log/collector/collector_logger.txt"
    cnn_log = "game_2048_npct-2_stochastic_muzero_ns100_upc200_rer0.0_bs512_chance-True_sslw2_seed0_250729_140944/log/collector/collector_logger.txt"

    print("="*100)
    print("GAT vs CNN-based Stochastic MuZero - Detailed Performance Comparison")
    print("="*100)
    print()

    # データ読み込み
    print("Loading log files...")
    gat_data = parse_full_log(gat_log)
    cnn_data = parse_full_log(cnn_log)

    print(f"  GAT: {len(gat_data)} collection episodes")
    print(f"  CNN: {len(cnn_data)} collection episodes")
    print()

    if not gat_data or not cnn_data:
        print("Error: Could not parse log files")
        return

    # 最大ステップ数を取得
    gat_max_steps = int(gat_data[-1]['total_envstep_count']) if gat_data else 0
    cnn_max_steps = int(cnn_data[-1]['total_envstep_count']) if cnn_data else 0

    print(f"Total Steps:")
    print(f"  GAT: {gat_max_steps:,} steps")
    print(f"  CNN: {cnn_max_steps:,} steps")
    print()

    # 共通のステップ数で比較
    common_steps = min(gat_max_steps, cnn_max_steps)
    print(f"Comparing at: {common_steps:,} steps (common baseline)")
    print()

    # 指定ステップ数までのデータをフィルタ
    gat_filtered = filter_by_steps(gat_data, common_steps)
    cnn_filtered = filter_by_steps(cnn_data, common_steps)

    print(f"Episodes analyzed:")
    print(f"  GAT: {len(gat_filtered)} episodes")
    print(f"  CNN: {len(cnn_filtered)} episodes")
    print()

    # 全体統計
    gat_stats = compute_stats(gat_filtered)
    cnn_stats = compute_stats(cnn_filtered)

    # 最後の10エピソードの統計
    gat_recent = compute_stats(gat_filtered, last_n=10)
    cnn_recent = compute_stats(cnn_filtered, last_n=10)

    print("="*100)
    print(f"OVERALL STATISTICS (All Episodes up to {common_steps:,} steps)")
    print("="*100)
    print()

    print("--- GAT Model (Graph Attention Network) ---")
    print(f"  Total Episodes:          {int(gat_stats.get('total_episode_count', 0)):,}")
    print(f"  Total Steps:             {int(gat_stats.get('total_envstep_count', 0)):,}")
    print(f"  Total Duration:          {gat_stats.get('total_duration', 0):.1f}s ({gat_stats.get('total_duration', 0)/60:.1f} min)")
    print(f"  Average Reward:          {gat_stats.get('reward_mean_avg', 0):.2f} +/- {gat_stats.get('reward_mean_std', 0):.2f}")
    print(f"  Max Reward (Peak):       {gat_stats.get('reward_max_max', 0):.1f}")
    print(f"  Avg Steps/Episode:       {gat_stats.get('avg_envstep_per_episode_avg', 0):.2f}")
    print(f"  Avg Speed:               {gat_stats.get('avg_envstep_per_sec_avg', 0):.2f} steps/sec")
    print(f"  Avg Collection Time:     {gat_stats.get('collect_time_avg', 0):.2f}s per collection")
    print()

    print("--- CNN Model (Baseline) ---")
    print(f"  Total Episodes:          {int(cnn_stats.get('total_episode_count', 0)):,}")
    print(f"  Total Steps:             {int(cnn_stats.get('total_envstep_count', 0)):,}")
    print(f"  Total Duration:          {cnn_stats.get('total_duration', 0):.1f}s ({cnn_stats.get('total_duration', 0)/60:.1f} min)")
    print(f"  Average Reward:          {cnn_stats.get('reward_mean_avg', 0):.2f} +/- {cnn_stats.get('reward_mean_std', 0):.2f}")
    print(f"  Max Reward (Peak):       {cnn_stats.get('reward_max_max', 0):.1f}")
    print(f"  Avg Steps/Episode:       {cnn_stats.get('avg_envstep_per_episode_avg', 0):.2f}")
    print(f"  Avg Speed:               {cnn_stats.get('avg_envstep_per_sec_avg', 0):.2f} steps/sec")
    print(f"  Avg Collection Time:     {cnn_stats.get('collect_time_avg', 0):.2f}s per collection")
    print()

    print("="*100)
    print("COMPARISON (GAT vs CNN) - Overall Performance")
    print("="*100)
    print()

    # 報酬比較
    reward_diff = gat_stats['reward_mean_avg'] - cnn_stats['reward_mean_avg']
    reward_pct = (reward_diff / cnn_stats['reward_mean_avg']) * 100

    max_reward_diff = gat_stats['reward_max_max'] - cnn_stats['reward_max_max']
    max_reward_pct = (max_reward_diff / cnn_stats['reward_max_max']) * 100

    # 速度比較
    speed_diff = gat_stats['avg_envstep_per_sec_avg'] - cnn_stats['avg_envstep_per_sec_avg']
    speed_pct = (speed_diff / cnn_stats['avg_envstep_per_sec_avg']) * 100

    # 時間比較
    time_diff = gat_stats['total_duration'] - cnn_stats['total_duration']
    time_pct = (time_diff / cnn_stats['total_duration']) * 100

    # ステップ/エピソード比較
    steps_per_ep_diff = gat_stats['avg_envstep_per_episode_avg'] - cnn_stats['avg_envstep_per_episode_avg']
    steps_per_ep_pct = (steps_per_ep_diff / cnn_stats['avg_envstep_per_episode_avg']) * 100

    print("Performance Metrics:")
    print(f"  Average Reward:           GAT: {gat_stats['reward_mean_avg']:.2f}  |  CNN: {cnn_stats['reward_mean_avg']:.2f}")
    print(f"  Difference:               {reward_diff:+.2f} ({reward_pct:+.2f}%)")
    print(f"  Winner:                   {'[+] GAT is BETTER' if reward_diff > 0 else '[-] CNN is BETTER'}")
    print()

    print("Peak Performance:")
    print(f"  Max Reward:               GAT: {gat_stats['reward_max_max']:.1f}  |  CNN: {cnn_stats['reward_max_max']:.1f}")
    print(f"  Difference:               {max_reward_diff:+.1f} ({max_reward_pct:+.2f}%)")
    print(f"  Winner:                   {'[+] GAT is BETTER' if max_reward_diff > 0 else '[-] CNN is BETTER'}")
    print()

    print("Execution Speed:")
    print(f"  Avg Steps/Second:         GAT: {gat_stats['avg_envstep_per_sec_avg']:.2f}  |  CNN: {cnn_stats['avg_envstep_per_sec_avg']:.2f}")
    print(f"  Difference:               {speed_diff:+.2f} ({speed_pct:+.2f}%)")
    print(f"  Winner:                   {'[+] GAT is FASTER' if speed_diff > 0 else '[-] CNN is FASTER'}")
    print()

    print("Total Time:")
    print(f"  Total Duration:           GAT: {gat_stats['total_duration']:.1f}s  |  CNN: {cnn_stats['total_duration']:.1f}s")
    print(f"  Difference:               {time_diff:+.1f}s ({time_pct:+.2f}%)")
    print(f"  Winner:                   {'[+] GAT is FASTER' if time_diff < 0 else '[-] CNN is FASTER'}")
    print()

    print("Episode Length:")
    print(f"  Avg Steps/Episode:        GAT: {gat_stats['avg_envstep_per_episode_avg']:.2f}  |  CNN: {cnn_stats['avg_envstep_per_episode_avg']:.2f}")
    print(f"  Difference:               {steps_per_ep_diff:+.2f} ({steps_per_ep_pct:+.2f}%)")
    print(f"  Comment:                  {'Episodes are longer (better policy)' if steps_per_ep_diff > 0 else 'Episodes are shorter'}")
    print()

    print("="*100)
    print("RECENT PERFORMANCE (Last 10 Episodes)")
    print("="*100)
    print()

    reward_recent_diff = gat_recent['reward_mean_avg'] - cnn_recent['reward_mean_avg']
    reward_recent_pct = (reward_recent_diff / cnn_recent['reward_mean_avg']) * 100

    print(f"  GAT Recent Avg Reward:    {gat_recent.get('reward_mean_avg', 0):.2f} +/- {gat_recent.get('reward_mean_std', 0):.2f}")
    print(f"  CNN Recent Avg Reward:    {cnn_recent.get('reward_mean_avg', 0):.2f} +/- {cnn_recent.get('reward_mean_std', 0):.2f}")
    print(f"  Difference:               {reward_recent_diff:+.2f} ({reward_recent_pct:+.2f}%)")
    print()

    print("="*100)
    print("FINAL SUMMARY")
    print("="*100)
    print()

    gat_wins = 0
    cnn_wins = 0
    metrics = []

    if reward_diff > 0:
        metrics.append(f"[+] GAT achieves {abs(reward_pct):.1f}% HIGHER average reward")
        gat_wins += 1
    else:
        metrics.append(f"[-] CNN achieves {abs(reward_pct):.1f}% higher average reward")
        cnn_wins += 1

    if max_reward_diff > 0:
        metrics.append(f"[+] GAT reaches {abs(max_reward_pct):.1f}% HIGHER peak reward")
        gat_wins += 1
    else:
        metrics.append(f"[-] CNN reaches {abs(max_reward_pct):.1f}% higher peak reward")
        cnn_wins += 1

    if speed_diff > 0:
        metrics.append(f"[+] GAT is {abs(speed_pct):.1f}% FASTER in execution speed")
        gat_wins += 1
    else:
        metrics.append(f"[-] CNN is {abs(speed_pct):.1f}% faster in execution speed")
        cnn_wins += 1

    if time_diff < 0:
        metrics.append(f"[+] GAT completes in {abs(time_pct):.1f}% LESS total time")
        gat_wins += 1
    else:
        metrics.append(f"[-] CNN completes in {abs(time_pct):.1f}% less total time")
        cnn_wins += 1

    if steps_per_ep_diff > 0:
        metrics.append(f"[+] GAT episodes are {abs(steps_per_ep_pct):.1f}% LONGER (better survival)")
        gat_wins += 1
    else:
        metrics.append(f"[-] CNN episodes are {abs(steps_per_ep_pct):.1f}% longer")
        cnn_wins += 1

    for metric in metrics:
        print(f"  {metric}")

    print()
    print(f"Overall Score: GAT {gat_wins} - {cnn_wins} CNN")
    print()

    if gat_wins > cnn_wins:
        print("*** WINNER: GAT (Graph Attention Network) ***")
        print(f"    GAT outperforms CNN in {gat_wins} out of 5 key metrics")
    elif cnn_wins > gat_wins:
        print("*** WINNER: CNN (Baseline) ***")
        print(f"    CNN outperforms GAT in {cnn_wins} out of 5 key metrics")
    else:
        print("*** TIE: Both models perform equally ***")

    print()
    print("="*100)


if __name__ == "__main__":
    main()
