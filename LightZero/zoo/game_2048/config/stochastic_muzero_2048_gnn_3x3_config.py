"""
Configuration for GNN-based Stochastic MuZero on 2048 (3x3 grid)
Uses GraphSAGE instead of CNN for state representation

【プログラム概要】
このファイルは2048ゲーム（3×3版）をGNN（Graph Neural Network: グラフニューラルネットワーク）と
Stochastic MuZero（確率的ムゼロ）アルゴリズムで学習させる設定ファイルです。

【全体の動作フロー】
1. 環境とモデル初期化
2. データ収集フェーズ: 8並列環境でMCTS（モンテカルロ木探索）実行
3. リプレイバッファに保存: 経験データを保存
4. 学習フェーズ: バッチサイズ512で200回更新
5. 評価フェーズ: 2000ステップごとに性能測定
6. 100万ステップに達するまで2〜5を繰り返し
"""
from easydict import EasyDict


# ==============================================================
# GNN-specific hyperparameters for 3x3 grid
# GNN特有のハイパーパラメータ（3×3グリッド用）
# ==============================================================

# 基本設定
env_id = 'game_2048'  # 環境ID: 2048ゲーム
grid_size = 3  # グリッドサイズ: 3×3のボード
action_space_size = 4  # アクションスペースサイズ: 上下左右の4方向
use_ture_chance_label_in_chance_encoder = True  # チャンスエンコーダで真のラベルを使用

# 環境とトレーニングの設定
collector_env_num = 8  # コレクター環境数: データ収集用に8つの並列環境
n_episode = 8  # エピソード数: 1回の収集で8エピソード
evaluator_env_num = 3  # 評価環境数: 性能評価用に3つの並列環境
num_simulations = 100  # シミュレーション数: MCTS（モンテカルロ木探索）で100回シミュレーション
update_per_collect = 200  # 収集ごとの更新回数: データ収集後に200回モデルを更新
batch_size = 512  # バッチサイズ: 一度に512サンプルで学習
max_env_step = int(1e6)  # 最大環境ステップ: 100万ステップで終了
reanalyze_ratio = 0.0  # 再解析比率: 過去データの再解析（0なので無効）
num_of_possible_chance_tile = 2  # 可能なチャンスタイル数: 2（タイル2か4）
chance_space_size = (grid_size * grid_size) * num_of_possible_chance_tile  # チャンス空間サイズ: 9*2=18 for 3x3

# GNN（グラフニューラルネットワーク）のハイパーパラメータ
num_gnn_layers = 3  # GNN層数: 3層のグラフ畳み込み
gnn_hidden_dim = 128  # GNN隠れ層次元: 128次元の特徴ベクトル
include_row_col_edges = True  # 行・列エッジを含む: 長距離接続を追加（同じ行・列のノード間も接続）
gnn_dropout = 0.0  # ドロップアウト率: 過学習防止（0なので無効）
# ==============================================================
# End of GNN-specific config
# GNN特有の設定ここまで
# ==============================================================

# メイン設定辞書
game_2048_gnn_stochastic_muzero_config = dict(
    # 実験名: データ保存先のパスとして使用
    exp_name=f'data_gnn_stochastic_mz_3x3/game_2048_gnn_3x3_npct-{num_of_possible_chance_tile}_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_bs{batch_size}_gnn{num_gnn_layers}L{gnn_hidden_dim}D_seed0',
    
    # 環境設定
    env=dict(
        stop_value=int(1e6),  # 停止値: 最大スコア100万で終了
        env_id=env_id,  # 環境ID
        obs_shape=(16, grid_size, grid_size),  # 観測形状: 16チャンネル×3×3のテンソル（各タイル値を1チャンネルで表現）
        grid_size=grid_size,  # グリッドサイズを明示的に指定
        obs_type='dict_encoded_board',  # 観測タイプ: エンコードされたボード表現
        num_of_possible_chance_tile=num_of_possible_chance_tile,  # 可能なチャンスタイル数
        collector_env_num=collector_env_num,  # データ収集用の並列環境数
        evaluator_env_num=evaluator_env_num,  # 評価用の並列環境数
        n_evaluator_episode=evaluator_env_num,  # 評価エピソード数
        manager=dict(shared_memory=False, ),  # 環境マネージャー設定: 共有メモリは使用しない
    ),
    # ポリシー（方策）設定
    policy=dict(
        # モデル構造の設定
        model=dict(
            observation_shape=(16, grid_size, grid_size),  # 観測形状: 3×3に変更
            action_space_size=action_space_size,  # アクション空間サイズ: 4方向
            chance_space_size=chance_space_size,  # チャンス空間サイズ: 18（9マス×2タイル）
            model_type='gnn',  # モデルタイプ: GNN（従来のCNNではなく）
            image_channel=16,  # 観測のチャンネル数: 16（各タイル値を1チャンネルで表現）
            frame_stack_num=1,  # フレームスタック数: 1フレームのみ使用
            
            # GNN特有のパラメータ
            num_channels=gnn_hidden_dim,  # チャンネル数: 128次元
            num_gnn_layers=num_gnn_layers,  # GNN層数: 3層
            grid_size=grid_size,  # グリッドサイズ: 3×3
            include_row_col_edges=include_row_col_edges,  # 行・列エッジを含む
            dropout=gnn_dropout,  # ドロップアウト率
            
            # 各ヘッド（予測器）の隠れ層設定
            value_head_hidden_channels=[128, 64],  # 価値ヘッド: 状態の良さを予測（128→64層）
            policy_head_hidden_channels=[128, 64],  # 方策ヘッド: どの行動が良いか予測（128→64層）
            reward_head_hidden_channels=[128, 64],  # 報酬ヘッド: 報酬を予測（128→64層）
            
            # サポートサイズ（Categorical Distributionのためのビン数）
            reward_support_size=601,  # 報酬サポートサイズ: -300〜+300の範囲を601個に離散化
            value_support_size=601,  # 価値サポートサイズ: 同様に離散化
            categorical_distribution=True,  # カテゴリカル分布: 期待値だけでなく分布全体を学習
            
            # SSL（Self-Supervised Learning: 自己教師あり学習）設定
            self_supervised_learning_loss=True,  # SSL損失を使用: 補助タスクで表現学習を強化
            proj_hid=1024,  # 射影層の隠れ層次元
            proj_out=1024,  # 射影層の出力次元
            pred_hid=512,  # 予測層の隠れ層次元
            pred_out=1024,  # 予測層の出力次元
            
            # その他の設定
            last_linear_layer_init_zero=True,  # 最終線形層をゼロ初期化: 学習初期の安定化
        ),
        # 事前学習済みモデルのパス
        model_path=None,  # Noneの場合はランダム初期化
        use_ture_chance_label_in_chance_encoder=use_ture_chance_label_in_chance_encoder,  # チャンスエンコーダで真のラベルを使用
        cuda=True,  # GPU使用: CUDAを有効化
        env_type='not_board_games',  # 環境タイプ: ボードゲーム以外
        game_segment_length=200,  # ゲームセグメント長: 1エピソードあたりの最大ステップ数
        
        # 学習設定
        update_per_collect=update_per_collect,  # 収集ごとの更新回数: 200回
        batch_size=batch_size,  # バッチサイズ: 512サンプル
        learning_rate=0.003,  # 学習率: 勾配降下法のステップサイズ
        grad_clip_value=0.5,  # 勾配クリッピング値: 勾配爆発を防ぐ（勾配の最大値を0.5に制限）
        
        # MCTS設定
        num_simulations=num_simulations,  # MCTSシミュレーション回数: 100回
        reanalyze_ratio=reanalyze_ratio,  # 再解析比率: 過去データの再解析（0なので無効）
        
        # 損失の重み
        ssl_loss_weight=2,  # SSL損失の重み: 自己教師あり学習の損失に2倍の重みをかける
        
        # データ収集と評価
        n_episode=n_episode,  # エピソード数: 1回の収集で8エピソード
        eval_freq=int(2e3),  # 評価頻度: 2000環境ステップごとに性能評価
        replay_buffer_size=int(1e6),  # リプレイバッファサイズ: 100万ステップ分のデータを保存
        collector_env_num=collector_env_num,  # コレクター環境数
        evaluator_env_num=evaluator_env_num,  # 評価環境数
    ),
)

# EasyDict化: ドット記法でアクセス可能にする（例: config.env.grid_size）
game_2048_gnn_stochastic_muzero_config = EasyDict(game_2048_gnn_stochastic_muzero_config)
main_config = game_2048_gnn_stochastic_muzero_config

# 作成設定: 環境とポリシーのインスタンス化に必要な情報
game_2048_gnn_stochastic_muzero_create_config = dict(
    env=dict(
        type='game_2048',  # 環境タイプ: 2048ゲーム
        import_names=['zoo.game_2048.envs.game_2048_env'],  # インポートするモジュール名
    ),
    env_manager=dict(type='subprocess'),  # 環境マネージャー: サブプロセスで並列実行
    policy=dict(
        type='stochastic_muzero',  # ポリシータイプ: Stochastic MuZero（確率的ムゼロ）
        import_names=['lzero.policy.stochastic_muzero'],  # インポートするモジュール名
    ),
)
# EasyDict化
game_2048_gnn_stochastic_muzero_create_config = EasyDict(game_2048_gnn_stochastic_muzero_create_config)
create_config = game_2048_gnn_stochastic_muzero_create_config

if __name__ == "__main__":
    """
    【メイン実行部分】
    このスクリプトを直接実行した場合のエントリーポイント
    
    実行時の動作フロー:
    1. ログ設定を初期化
    2. 乱数シードを設定（再現性確保）
    3. train_muzero関数を呼び出して学習開始
    
    Note:
        このスクリプトは ``python3 -u`` で実行すべきです（unbuffered: バッファリングなし出力）
    """
    import logging
    import sys
    from ding.utils import set_pkg_seed
    
    # ログ設定: INFO レベル以上のログを標準出力に表示
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s](%(filename)s:%(lineno)d): %(message)s',
        stream=sys.stdout,
    )
    
    # エントリーポイント: MuZero学習ループの開始
    from lzero.entry import train_muzero
    
    # 乱数シード設定: 再現性を確保（同じシードなら同じ結果が得られる）
    set_pkg_seed(0)
    
    # 学習開始
    # - [main_config, create_config]: 設定辞書のリスト
    # - seed=0: 乱数シード
    # - model_path: 事前学習済みモデルのパス（Noneの場合はランダム初期化）
    # - max_env_step: 最大環境ステップ数（100万ステップ）
    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)
