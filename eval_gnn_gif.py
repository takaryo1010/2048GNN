"""
GIF動画出力スクリプト
ゲームプレイをGIF形式で保存します（ファイルサイズが小さい）
"""
import numpy as np
import os
import sys

# LightZeroのパスを追加
sys.path.append('./LightZero')

from lzero.entry import eval_muzero
from zoo.game_2048.config.stochastic_muzero_2048_gnn_config import main_config, create_config

if __name__ == "__main__":
    # トレーニング済みモデルのパス
    model_path = './LightZero/zoo/game_2048/config/data_gnn_stochastic_mz/game_2048_gnn_npct-2_ns100_upc200_rer0.0_bs512_gnn3L128D_sparse_seed0_251008_050852/ckpt/ckpt_best.pth.tar'
    
    # 設定
    seeds = [0, 1, 2]  # 3つのエピソード
    num_episodes_each_seed = 1
    
    # GIF出力設定
    replay_path = './gif_output'
    replay_format = 'gif'  # GIF形式
    
    # 環境設定
    main_config.env.render_mode = 'image_savefile_mode'
    main_config.env.replay_path = replay_path
    main_config.env.replay_format = replay_format
    main_config.env.replay_name_suffix = 'gnn_2048'
    main_config.env.max_episode_steps = int(1e9)
    main_config.env.ignore_legal_actions = False  # 正しく終了判定を行う
    
    total_episodes = len(seeds) * num_episodes_each_seed
    create_config.env_manager.type = 'base'
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = total_episodes
    
    print("=" * 60)
    print("GNN 2048 - GIF動画出力")
    print("=" * 60)
    print(f"エピソード数: {total_episodes}")
    print(f"出力先: {replay_path}")
    print("=" * 60)
    
    # 出力ディレクトリの作成
    os.makedirs(replay_path, exist_ok=True)
    
    all_returns = []
    
    # 各シードで評価
    for seed in seeds:
        print(f"\nエピソード {seed+1}/{total_episodes} を実行中...")
        
        returns_mean, returns = eval_muzero(
            [main_config, create_config],
            seed=seed,
            num_episodes_each_seed=num_episodes_each_seed,
            print_seed_details=True,
            model_path=model_path
        )
        
        all_returns.append(float(returns[0][0]))
        print(f"  報酬: {float(returns[0][0]):.2f}")
    
    print("\n" + "=" * 60)
    print("完了!")
    print("=" * 60)
    print(f"平均報酬: {np.mean(all_returns):.2f}")
    print(f"最高報酬: {np.max(all_returns):.2f}")
    print(f"GIF保存先: {replay_path}")
    print("=" * 60)
