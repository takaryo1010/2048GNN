"""
MP4動画出力スクリプト for GNN-based Stochastic MuZero on 2048
トレーニング済みモデルを使ってゲームプレイをMP4形式で保存します
"""
import numpy as np
import os
import sys

# LightZeroのパスを追加
sys.path.append('./LightZero')

from lzero.entry import eval_muzero
from zoo.game_2048.config.stochastic_muzero_2048_gnn_config import main_config, create_config

if __name__ == "__main__":
    """
    トレーニング済みGNNモデルを使って2048ゲームプレイをMP4動画として出力します
    """
    
    # ========================================
    # 設定 - 必要に応じて変更してください
    # ========================================
    
    # トレーニング済みモデルのパス
    model_path = './LightZero/zoo/game_2048/config/data_gnn_stochastic_mz/game_2048_gnn_npct-2_ns100_upc200_rer0.0_bs512_gnn3L128D_sparse_seed0_251008_050852/ckpt/ckpt_best.pth.tar'
    
    # 評価設定
    seeds = [0, 1, 2]  # テストするランダムシード
    num_episodes_each_seed = 5  # 各シードでのエピソード数
    
    # 動画出力設定
    render_mode = 'image_savefile_mode'  # 画像ファイル保存モード
    replay_path = './videos_gnn_output'  # 動画保存ディレクトリ
    replay_format = 'mp4'  # MP4形式で出力
    replay_name_suffix = 'gnn_muzero_eval'
    
    # ========================================
    # 環境設定
    # ========================================
    
    # レンダリング有効化
    main_config.env.render_mode = render_mode
    main_config.env.replay_path = replay_path
    main_config.env.replay_format = replay_format
    main_config.env.replay_name_suffix = replay_name_suffix
    
    # ゲームが自然に終了するまでプレイ
    main_config.env.max_episode_steps = int(1e9)
    main_config.env.ignore_legal_actions = False  # 正しく終了判定を行う
    
    # 動画出力には base environment manager と単一環境が必要
    total_test_episodes = num_episodes_each_seed * len(seeds)
    create_config.env_manager.type = 'base'
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = total_test_episodes
    
    # ========================================
    # モデルの存在確認
    # ========================================
    
    if not os.path.exists(model_path):
        print(f"エラー: モデルファイルが見つかりません: {model_path}")
        print(f"\n利用可能なチェックポイントを探しています...")
        
        # チェックポイントディレクトリを探す
        config_dir = './LightZero/zoo/game_2048/config/data_gnn_stochastic_mz'
        if os.path.exists(config_dir):
            print(f"\n{config_dir} の内容:")
            for item in os.listdir(config_dir):
                item_path = os.path.join(config_dir, item)
                if os.path.isdir(item_path):
                    ckpt_dir = os.path.join(item_path, 'ckpt')
                    if os.path.exists(ckpt_dir):
                        print(f"\n  ディレクトリ: {item}")
                        ckpts = os.listdir(ckpt_dir)
                        for ckpt in ckpts:
                            print(f"    - {ckpt}")
        sys.exit(1)
    
    # ========================================
    # 評価実行
    # ========================================
    
    print("=" * 80)
    print("GNN-based Stochastic MuZero - 2048 MP4動画出力")
    print("=" * 80)
    print(f"モデルパス: {model_path}")
    print(f"シード: {seeds}")
    print(f"各シードのエピソード数: {num_episodes_each_seed}")
    print(f"総エピソード数: {total_test_episodes}")
    print(f"レンダーモード: {render_mode}")
    print(f"出力パス: {replay_path}")
    print(f"出力フォーマット: {replay_format}")
    print("=" * 80)
    
    # 出力ディレクトリの作成
    if not os.path.exists(replay_path):
        os.makedirs(replay_path)
        print(f"出力ディレクトリを作成しました: {replay_path}")
    
    returns_mean_seeds = []
    returns_seeds = []
    
    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"シード {seed} を評価中...")
        print(f"{'='*80}")
        
        returns_mean, returns = eval_muzero(
            [main_config, create_config],
            seed=seed,
            num_episodes_each_seed=num_episodes_each_seed,
            print_seed_details=True,
            model_path=model_path
        )
        
        print(f"\nシード {seed} の結果:")
        print(f"  平均リターン: {returns_mean:.2f}")
        print(f"  エピソード別リターン: {returns}")
        
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)
    
    # ========================================
    # 統計サマリー
    # ========================================
    
    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)
    
    print("\n" + "=" * 80)
    print("評価サマリー")
    print("=" * 80)
    print(f"評価したシード数: {len(seeds)}")
    print(f"各シードのエピソード数: {num_episodes_each_seed}")
    print(f"テストしたシード: {seeds}")
    print(f"\n各シードの平均リターン: {returns_mean_seeds}")
    print(f"全エピソードのリターン形状: {returns_seeds.shape}")
    print(f"\n全体の平均報酬: {returns_mean_seeds.mean():.2f}")
    print(f"全体の標準偏差: {returns_mean_seeds.std():.2f}")
    print(f"最高エピソード報酬: {float(np.max(returns_seeds)):.2f}")
    print(f"最低エピソード報酬: {float(np.min(returns_seeds)):.2f}")
    print("=" * 80)
    print(f"\nMP4動画ファイルが保存されました: {replay_path}")
    print("=" * 80)
