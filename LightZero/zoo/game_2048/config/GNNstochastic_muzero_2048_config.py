from easydict import EasyDict
from lzero.model.stochastic_muzero_model_gnn import StochasticMuZeroModelGNN
# ==============================================================
# begin of the most frequently changed config specified by the user
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
max_env_step = int(1e9)
reanalyze_ratio = 0.
num_of_possible_chance_tile = 2
chance_space_size = 16 * num_of_possible_chance_tile # 4x4盤面の場合
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

game_2048_stochastic_muzero_config = dict(
    exp_name=f'data_stochastic_mz_gnn/game_2048_gnn_ns{num_simulations}_upc{update_per_collect}_bs{batch_size}_seed0',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        obs_shape=(16, 4, 4),
        obs_type='dict_encoded_board',
        num_of_possible_chance_tile=num_of_possible_chance_tile,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            type='StochasticMuZeroModelGNN',
            observation_shape=(16, 4, 4),
            action_space_size=action_space_size,
            chance_space_size=chance_space_size,
            node_embed_dim=64,
            gnn_hidden_dim=128,
            gnn_num_layers=3,
            latent_state_dim=256,
            mlp_hidden_dim=256,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
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
        ssl_loss_weight=2,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
game_2048_stochastic_muzero_config = EasyDict(game_2048_stochastic_muzero_config)
main_config = game_2048_stochastic_muzero_config

game_2048_stochastic_muzero_create_config = dict(
    env=dict(
        type='game_2048',
        import_names=['zoo.game_2048.envs.game_2048_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='stochastic_muzero',
        import_names=['lzero.policy.stochastic_muzero'],
    ),
    # ▼▼▼ モデルのインポートパスを指定 ▼▼▼
    model=dict(
        type='StochasticMuZeroModelGNN',
        import_names=['zoo.game_2048.model.stochastic_muzero_model_gnn'],
    )
)
game_2048_stochastic_muzero_create_config = EasyDict(game_2048_stochastic_muzero_create_config)
create_config = game_2048_stochastic_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    import torch
    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)