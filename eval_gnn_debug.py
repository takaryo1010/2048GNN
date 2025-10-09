"""
デバッグ用: ゲーム終了判定を詳しくログ出力
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
    
    # デバッグ設定
    seeds = [0]
    num_episodes_each_seed = 1
    
    # 動画出力設定
    replay_path = './video_debug'
    replay_format = 'mp4'
    
    # 環境設定
    main_config.env.render_mode = 'image_savefile_mode'
    main_config.env.replay_path = replay_path
    main_config.env.replay_format = replay_format
    main_config.env.replay_name_suffix = 'debug'
    main_config.env.max_episode_steps = int(1e9)  # 無制限
    
    # ignore_legal_actionsをFalseに設定（正しく終了判定するため）
    main_config.env.ignore_legal_actions = False
    
    create_config.env_manager.type = 'base'
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    
    print("=" * 60)
    print("デバッグモード: ゲーム終了判定を詳細に確認")
    print("=" * 60)
    print(f"ignore_legal_actions: {main_config.env.ignore_legal_actions}")
    print(f"max_episode_steps: {main_config.env.max_episode_steps}")
    print("=" * 60)
    
    # 出力ディレクトリの作成
    os.makedirs(replay_path, exist_ok=True)
    
    # 評価実行
    returns_mean, returns = eval_muzero(
        [main_config, create_config],
        seed=0,
        num_episodes_each_seed=1,
        print_seed_details=True,
        model_path=model_path
    )
    
    print("\n" + "=" * 60)
    print("完了!")
    print("=" * 60)
    print(f"エピソードの報酬: {float(returns[0][0]):.2f}")
    print(f"動画保存先: {replay_path}")
    print("=" * 60)
