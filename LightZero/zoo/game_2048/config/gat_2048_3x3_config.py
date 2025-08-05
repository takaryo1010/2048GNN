from easydict import EasyDict


# ==============================================================
# 3x3 board configuration for GAT-based MuZero on 2048
# ==============================================================
env_id = 'game_2048'
action_space_size = 4
use_ture_chance_label_in_chance_encoder = True
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 100
update_per_collect = 200
batch_size = 512
max_env_step = int(1e6)  # Reduced for faster training
reanalyze_ratio = 0.
num_of_possible_chance_tile = 2
chance_space_size = 16 * num_of_possible_chance_tile

# GAT-specific configurations for 3x3 board
grid_size = 3  # 3x3 board
num_heads = 4
hidden_channels = 64
num_gat_layers = 3
state_dim = 256
dropout = 0.1
# ==============================================================

game_2048_gat_3x3_config = dict(
    exp_name=f'data_gat_3x3/game_2048_grid{grid_size}_gat_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_bs{batch_size}_heads{num_heads}_seed0',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        obs_shape=(16, grid_size, grid_size),  # 3x3 grid
        obs_type='dict_encoded_board',
        num_of_possible_chance_tile=num_of_possible_chance_tile,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # Additional environment config for 3x3 board
        board_size=grid_size,
    ),
    policy=dict(
        model=dict(
            model_type='gat',  # Use GAT model type
            observation_shape=(16, grid_size, grid_size),
            action_space_size=action_space_size,
            # GAT-specific parameters
            num_heads=num_heads,
            hidden_channels=hidden_channels,
            num_gat_layers=num_gat_layers,
            state_dim=state_dim,
            dropout=dropout,
            # Standard parameters
            value_head_channels=16,
            policy_head_channels=16,
            value_head_hidden_channels=[32],
            policy_head_hidden_channels=[32],
            reward_support_size=601,
            value_support_size=601,
            categorical_distribution=True,
            last_linear_layer_init_zero=True,
            state_norm=False,
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
        ssl_loss_weight=0,  # Disabled for GAT model
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
game_2048_gat_3x3_config = EasyDict(game_2048_gat_3x3_config)
main_config = game_2048_gat_3x3_config

game_2048_gat_3x3_create_config = dict(
    env=dict(
        type='game_2048',
        import_names=['zoo.game_2048.envs.game_2048_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
game_2048_gat_3x3_create_config = EasyDict(game_2048_gat_3x3_create_config)
create_config = game_2048_gat_3x3_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    import torch
    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    
    print(f"Training GAT-based model on {grid_size}x{grid_size} grid")
    print(f"GAT config: {num_heads} heads, {hidden_channels} hidden channels, {num_gat_layers} layers")
    
    # Import the new model to register it
    from lzero.model.gat_muzero_model import GATMuZeroModel
    print("Registered GATMuZeroModel")
    
    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)
