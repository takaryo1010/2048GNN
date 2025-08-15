from easydict import EasyDict
import os

# ==============================================================
# GAT StochasticMuZero 転移学習設定 (3×3 → 4×4)
# ==============================================================
# 1. まず3×3で学習
# 2. 学習済みモデルを4×4に転移
# ==============================================================

env_id = 'game_2048'
action_space_size = 4
use_ture_chance_label_in_chance_encoder = True

# 基本設定
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 100
update_per_collect = 200
batch_size = 512
max_env_step = int(1e6)
reanalyze_ratio = 0.
num_of_possible_chance_tile = 2

# GAT設定（転移学習で共通）
num_heads = 4
hidden_channels = 64
num_gat_layers = 3
state_dim = 256
dropout = 0.1

# StochasticMuZero specific parameters
chance_encoder_num_layers = 2
afterstate_reward_layers = 2
stochastic_loss_weight = 1.0

# ==============================================================
# PHASE 1: 3×3グリッドでの学習設定
# ==============================================================
def get_3x3_config():
    grid_size = 3
    chance_space_size = (grid_size ** 2) * num_of_possible_chance_tile  # 18
    
    config_3x3 = dict(
        exp_name=f'data_gat_transfer/phase1_3x3_gat_stochastic_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_bs{batch_size}_heads{num_heads}_seed0',
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            obs_shape=(16, grid_size, grid_size),
            obs_type='dict_encoded_board',
            num_of_possible_chance_tile=num_of_possible_chance_tile,
            collector_env_num=collector_env_num,
            evaluator_env_num=1,
            n_evaluator_episode=1,
            manager=dict(shared_memory=False, ),
        ),
        policy=dict(
            type='stochastic_muzero',
            model=dict(
                type='GATStochasticMuZeroModel',
                model_type='gat',
                model='gat_stochastic',
                observation_shape=(16, grid_size, grid_size),
                image_channel=16,
                action_space_size=action_space_size,
                chance_space_size=chance_space_size,
                frame_stack_num=1,
                
                # GAT parameters
                num_heads=num_heads,
                hidden_channels=hidden_channels,
                num_gat_layers=num_gat_layers,
                state_dim=state_dim,
                dropout=dropout,
                
                # StochasticMuZero parameters
                chance_encoder_num_layers=chance_encoder_num_layers,
                afterstate_reward_layers=afterstate_reward_layers,
                
                # Standard parameters
                value_head_channels=16,
                policy_head_channels=16,
                value_head_hidden_channels=[32],
                policy_head_hidden_channels=[32],
                
                flatten_input_size_for_value_head=state_dim,
                flatten_input_size_for_policy_head=state_dim,
                
                reward_support_size=601,
                value_support_size=601,
                categorical_distribution=True,
                last_linear_layer_init_zero=True,
                state_norm=False,
                self_supervised_learning_loss=True,
            ),
            model_path=None,
            use_ture_chance_label_in_chance_encoder=use_ture_chance_label_in_chance_encoder,
            cuda=True,
            game_segment_length=200,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            td_steps=10,
            discount_factor=0.999,
            manual_temperature_decay=True,
            optim_type='Adam',
            piecewise_decay_lr_scheduler=False,
            learning_rate=0.003,
            weight_decay=1e-4,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            ssl_loss_weight=0,
            stochastic_loss_weight=stochastic_loss_weight,
            n_episode=n_episode,
            eval_freq=int(2e5),  # 短めに設定して転移のタイミングを早める
            replay_buffer_size=int(1e6),
            collector_env_num=collector_env_num,
            evaluator_env_num=1,
        ),
    )
    return EasyDict(config_3x3)

# ==============================================================
# PHASE 2: 4×4グリッドへの転移学習設定
# ==============================================================
def get_4x4_transfer_config(pretrained_model_path):
    grid_size = 4
    chance_space_size = (grid_size ** 2) * num_of_possible_chance_tile  # 32
    
    config_4x4 = dict(
        exp_name=f'data_gat_transfer/phase2_4x4_transfer_gat_stochastic_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_bs{batch_size}_heads{num_heads}_seed0',
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            obs_shape=(16, grid_size, grid_size),
            obs_type='dict_encoded_board',
            num_of_possible_chance_tile=num_of_possible_chance_tile,
            collector_env_num=collector_env_num,
            evaluator_env_num=1,
            n_evaluator_episode=1,
            manager=dict(shared_memory=False, ),
        ),
        policy=dict(
            type='stochastic_muzero',
            model=dict(
                type='GATStochasticMuZeroModel',
                model_type='gat',
                model='gat_stochastic',
                observation_shape=(16, grid_size, grid_size),
                image_channel=16,
                action_space_size=action_space_size,
                chance_space_size=chance_space_size,
                frame_stack_num=1,
                
                # GAT parameters (同じ設定を維持)
                num_heads=num_heads,
                hidden_channels=hidden_channels,
                num_gat_layers=num_gat_layers,
                state_dim=state_dim,
                dropout=dropout,
                
                # StochasticMuZero parameters
                chance_encoder_num_layers=chance_encoder_num_layers,
                afterstate_reward_layers=afterstate_reward_layers,
                
                # Standard parameters
                value_head_channels=16,
                policy_head_channels=16,
                value_head_hidden_channels=[32],
                policy_head_hidden_channels=[32],
                
                flatten_input_size_for_value_head=state_dim,
                flatten_input_size_for_policy_head=state_dim,
                
                reward_support_size=601,
                value_support_size=601,
                categorical_distribution=True,
                last_linear_layer_init_zero=True,
                state_norm=False,
                self_supervised_learning_loss=True,
            ),
            model_path=pretrained_model_path,  # 3×3で学習済みのモデルを指定
            use_ture_chance_label_in_chance_encoder=use_ture_chance_label_in_chance_encoder,
            cuda=True,
            game_segment_length=200,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            td_steps=10,
            discount_factor=0.999,
            manual_temperature_decay=True,
            optim_type='Adam',
            piecewise_decay_lr_scheduler=False,
            learning_rate=0.001,  # 転移学習では学習率を下げる
            weight_decay=1e-4,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            ssl_loss_weight=0,
            stochastic_loss_weight=stochastic_loss_weight,
            n_episode=n_episode,
            eval_freq=int(1e6),
            replay_buffer_size=int(1e6),
            collector_env_num=collector_env_num,
            evaluator_env_num=1,
        ),
    )
    return EasyDict(config_4x4)

# Create config
create_config = dict(
    env=dict(
        type='game_2048',
        import_names=['zoo.game_2048.envs.game_2048_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='stochastic_muzero',
        import_names=['lzero.policy.stochastic_muzero'],
    ),
    model=dict(
        type='GATStochasticMuZeroModel',
        import_names=['lzero.model.gat_stochastic_muzero_model'],
    ),
)
create_config = EasyDict(create_config)

if __name__ == "__main__":
    from lzero.entry import train_muzero
    import torch
    import logging
    import glob
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("GAT STOCHASTIC MUZERO 転移学習")
    print("3×3 → 4×4 グリッド転移")
    print("=" * 80)
    
    # PHASE 1: 3×3で学習
    print("\n🔄 PHASE 1: 3×3グリッドでの学習開始")
    print("=" * 50)
    
    config_3x3 = get_3x3_config()
    print(f"✓ 3×3設定: chance_space_size={config_3x3.policy.model.chance_space_size}")
    print(f"✓ 実験名: {config_3x3.exp_name}")
    
    # 3×3での学習実行
    print("🚀 3×3グリッドでのトレーニング開始...")
    train_muzero([config_3x3, create_config], seed=0, model_path=None, max_env_step=int(5e5))  # 短めに設定
    
    # 学習済みモデルのパスを見つける
    print("\n🔍 学習済みモデルの検索...")
    model_search_pattern = f"./data_gat_transfer/phase1_3x3_**/ckpt_*.pth.tar"
    model_files = glob.glob(model_search_pattern, recursive=True)
    
    if not model_files:
        print("❌ 学習済みモデルが見つかりません。Phase 1が正常に完了していません。")
        exit(1)
    
    # 最新のモデルファイルを選択
    latest_model = max(model_files, key=os.path.getctime)
    print(f"✅ 学習済みモデル発見: {latest_model}")
    
    # PHASE 2: 4×4への転移学習
    print("\n🔄 PHASE 2: 4×4グリッドへの転移学習開始")
    print("=" * 50)
    
    config_4x4 = get_4x4_transfer_config(latest_model)
    print(f"✓ 4×4設定: chance_space_size={config_4x4.policy.model.chance_space_size}")
    print(f"✓ 実験名: {config_4x4.exp_name}")
    print(f"✓ プリトレインモデル: {latest_model}")
    
    # 転移学習のための特別な設定
    print("\n📋 転移学習の特別設定:")
    print(f"  - 学習率: {config_4x4.policy.learning_rate} (元の学習率より低下)")
    print(f"  - プリトレインモデル: あり")
    print(f"  - GAT設定: 3×3と同じ (転移可能)")
    
    # 4×4での転移学習実行
    print("🚀 4×4グリッドでの転移学習開始...")
    train_muzero([config_4x4, create_config], seed=0, model_path=latest_model, max_env_step=max_env_step)
    
    print("\n🎉 転移学習完了！")
    print("=" * 80)
