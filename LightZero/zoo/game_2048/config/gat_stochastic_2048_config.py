from easydict import EasyDict


# ==============================================================
# StochasticMuZero with GAT configuration for 2048
# ==============================================================
env_id = 'game_2048'
action_space_size = 4
use_ture_chance_label_in_chance_encoder = True  # Key for StochasticMuZero
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 100
update_per_collect = 200
batch_size = 512
max_env_step = int(1e6)
reanalyze_ratio = 0.
num_of_possible_chance_tile = 2
chance_space_size = 16 * num_of_possible_chance_tile  # 32 chance outcomes

# GAT-specific configurations for StochasticMuZero
grid_size = 4  # Can be changed to 3 for 3x3 boards
num_heads = 4
hidden_channels = 64
num_gat_layers = 3
state_dim = 256
dropout = 0.1

# StochasticMuZero specific parameters
chance_encoder_num_layers = 2  # Layers for chance encoding
afterstate_reward_layers = 2   # Layers for afterstate reward prediction
stochastic_loss_weight = 1.0   # Weight for stochastic loss components
# ==============================================================

game_2048_gat_stochastic_config = dict(
    exp_name=f'data_gat_stochastic/game_2048_grid{grid_size}_gat_stochastic_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_bs{batch_size}_heads{num_heads}_seed0',
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
        type='stochastic_muzero',  # Use standard stochastic_muzero type for compatibility
        model=dict(
            # StochasticMuZero model configuration  
            model_type='conv',  # Use conv type for compatibility with existing policy
            model='gat_stochastic',   # Use GAT-based StochasticMuZero
            observation_shape=(16, grid_size, grid_size),
            image_channel=16,
            action_space_size=action_space_size,
            chance_space_size=chance_space_size,  # Important for StochasticMuZero
            frame_stack_num=1,
            
            # GAT-specific parameters
            num_heads=num_heads,
            hidden_channels=hidden_channels,
            num_gat_layers=num_gat_layers,
            state_dim=state_dim,
            dropout=dropout,
            
            # StochasticMuZero specific parameters
            chance_encoder_num_layers=chance_encoder_num_layers,
            afterstate_reward_layers=afterstate_reward_layers,
            
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
            self_supervised_learning_loss=True,  # Enable SSL for obs_target_batch generation
        ),
        model_path=None,
        use_ture_chance_label_in_chance_encoder=use_ture_chance_label_in_chance_encoder,  # Critical for StochasticMuZero
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
        
        # StochasticMuZero specific loss weights
        ssl_loss_weight=0,  # Disabled for GAT model
        stochastic_loss_weight=stochastic_loss_weight,  # Weight for stochastic components
        
        n_episode=n_episode,
        eval_freq=int(1e6),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=1,
    ),
)
game_2048_gat_stochastic_config = EasyDict(game_2048_gat_stochastic_config)
main_config = game_2048_gat_stochastic_config

game_2048_gat_stochastic_create_config = dict(
    env=dict(
        type='game_2048',
        import_names=['zoo.game_2048.envs.game_2048_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='stochastic_muzero',  # Use standard stochastic_muzero type for compatibility
        import_names=['lzero.policy.stochastic_muzero'],
    ),
)
game_2048_gat_stochastic_create_config = EasyDict(game_2048_gat_stochastic_create_config)
create_config = game_2048_gat_stochastic_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero  # Use standard MuZero training entry with StochasticMuZero policy
    import torch
    
    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    
    print(f"Training GAT-based StochasticMuZero on {grid_size}x{grid_size} grid")
    print(f"GAT config: {num_heads} heads, {hidden_channels} hidden channels, {num_gat_layers} layers")
    print(f"StochasticMuZero config: chance_space_size={chance_space_size}, use_chance_encoder={use_ture_chance_label_in_chance_encoder}")
    
    # Import the new StochasticMuZero GAT model
    from lzero.model.gat_stochastic_muzero_model import GATStochasticMuZeroModel
    print("Registered GATStochasticMuZeroModel")
    
    # Verify GAT StochasticMuZero model is properly registered
    from lzero.model import GATStochasticMuZeroModel as ImportedGATStochasticModel
    print(f"GAT StochasticMuZero Model class loaded: {ImportedGATStochasticModel}")
    
    # Force the use of GAT StochasticMuZero model
    main_config.policy.model.model_type = 'conv'
    print("Using conv model_type for StochasticMuZero policy compatibility")
    
    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)
