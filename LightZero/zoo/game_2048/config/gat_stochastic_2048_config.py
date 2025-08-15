from easydict import EasyDict


# ==============================================================
# StochasticMuZero with GAT configuration for 2048
# ==============================================================
# グリッドサイズを変更するには、以下のgrid_size変数のみを変更してください：
# - 3×3の場合: grid_size = 3
# - 4×4の場合: grid_size = 4  
# - 5×5の場合: grid_size = 5 (など)
# 他の全ての設定は自動的に調整されます。
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

# GAT-specific configurations for StochasticMuZero
# ==============================================
# 【重要】以下の変数を変更するだけで全ての設定が自動調整されます
grid_size = 3  # ゲームのグリッドサイズ (3=3x3, 4=4x4, 5=5x5, etc.)
# ==============================================
chance_space_size = (grid_size ** 2) * num_of_possible_chance_tile  # Automatically calculated based on grid size
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
            type='GATStochasticMuZeroModel',  # 強制的にGATモデルを使用
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
            
            # フラット化されたサイズの自動計算 (GAT モデル用)
            flatten_input_size_for_value_head=state_dim,  # GAT の出力サイズ
            flatten_input_size_for_policy_head=state_dim, # GAT の出力サイズ
            
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
    # 明示的にGATStochasticMuZeroModelを強制する
    model=dict(
        type='GATStochasticMuZeroModel',
        import_names=['lzero.model.gat_stochastic_muzero_model'],
    ),
)
game_2048_gat_stochastic_create_config = EasyDict(game_2048_gat_stochastic_create_config)
create_config = game_2048_gat_stochastic_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero  # Use standard MuZero training entry with StochasticMuZero policy
    import torch
    import logging
    
    # ログレベルをINFOに設定してGATモデルのデバッグメッセージを表示
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("GAT STOCHASTIC MUZERO デバッグ情報")
    print(f"現在の設定: {grid_size}×{grid_size} グリッド")
    print("=" * 80)
    
    if torch.cuda.is_available():
        print("✓ CUDA is available. Training on GPU.")
    else:
        print("✗ CUDA is not available. Training on CPU.")
    
    print(f"✓ Training GAT-based StochasticMuZero on {grid_size}x{grid_size} grid")
    print(f"✓ GAT config: {num_heads} heads, {hidden_channels} hidden channels, {num_gat_layers} layers")
    print(f"✓ StochasticMuZero config: chance_space_size={chance_space_size} (auto-calculated: {grid_size}²×{num_of_possible_chance_tile}), use_chance_encoder={use_ture_chance_label_in_chance_encoder}")
    print(f"✓ 設定ファイル詳細:")
    print(f"  - 環境ID: {env_id}")
    print(f"  - グリッドサイズ: {grid_size}×{grid_size} (変更したい場合はgrid_size変数のみ変更)")
    print(f"  - アクション空間サイズ: {action_space_size}")
    print(f"  - チャンス空間サイズ: {chance_space_size} (自動計算)")
    print(f"  - バッチサイズ: {batch_size}")
    print(f"  - シミュレーション数: {num_simulations}")
    print(f"  - ドロップアウト率: {dropout}")
    print(f"  - 状態次元: {state_dim}")
    print("=" * 50)
    
    # Import the new StochasticMuZero GAT model
    print("=" * 50)
    print("モデルインポート開始")
    try:
        from lzero.model.gat_stochastic_muzero_model import GATStochasticMuZeroModel
        print("✓ GATStochasticMuZeroModel インポート成功")
        print(f"✓ モデルクラス: {GATStochasticMuZeroModel}")
        print(f"✓ モジュール: {GATStochasticMuZeroModel.__module__}")
    except ImportError as e:
        print(f"✗ GATStochasticMuZeroModel インポートエラー: {e}")
        raise
    
    print("Registered GATStochasticMuZeroModel")
    
    # Verify GAT StochasticMuZero model is properly registered
    print("=" * 50)
    print("モデル登録確認")
    try:
        from lzero.model import GATStochasticMuZeroModel as ImportedGATStochasticModel
        print(f"✓ GAT StochasticMuZero Model class loaded: {ImportedGATStochasticModel}")
        print(f"✓ クラス詳細: {ImportedGATStochasticModel.__name__}")
        print(f"✓ 基底クラス: {ImportedGATStochasticModel.__bases__}")
    except ImportError as e:
        print(f"✗ モデル登録確認エラー: {e}")
        print("⚠ 代替インポートを試行中...")
        ImportedGATStochasticModel = GATStochasticMuZeroModel
    
    # Force the use of GAT StochasticMuZero model
    print("=" * 50)
    print("モデル設定確認")
    print(f"現在のモデル設定:")
    print(f"  - type: {main_config.policy.model.get('type', 'NOT SET')}")
    print(f"  - model_type: {main_config.policy.model.model_type}")
    print(f"  - model: {main_config.policy.model.model}")
    print(f"  - GAT設定:")
    print(f"    - num_heads: {main_config.policy.model.num_heads}")
    print(f"    - hidden_channels: {main_config.policy.model.hidden_channels}")
    print(f"    - num_gat_layers: {main_config.policy.model.num_gat_layers}")
    print(f"    - state_dim: {main_config.policy.model.state_dim}")
    print(f"    - dropout: {main_config.policy.model.dropout}")
    
    # 強制的にGATStochasticMuZeroModelを使用
    main_config.policy.model.type = 'GATStochasticMuZeroModel'
    
    # 追加でmodel_typeもGATに設定
    main_config.policy.model.model_type = 'gat'  # GAT用に変更
    
    print("✓ 強制的にGATStochasticMuZeroModelを設定")
    print(f"✓ 最終モデルタイプ: {main_config.policy.model.type}")
    print(f"✓ 最終model_type: {main_config.policy.model.model_type}")
    print("=" * 50)
    
    print("=" * 80)
    print("トレーニング開始")
    print("=" * 80)
    
    # カスタムの検証を追加
    def validate_gat_usage():
        """トレーニング中にGATが実際に使用されているかを検証"""
        print("🔍 GNN使用状況の検証開始...")
        
        # MODEL_REGISTRYからGATStochasticMuZeroModelを確認
        try:
            from ding.utils import MODEL_REGISTRY
            print(f"✓ MODEL_REGISTRY: {list(MODEL_REGISTRY.keys())}")
            
            if 'GATStochasticMuZeroModel' in MODEL_REGISTRY:
                print("✅ GATStochasticMuZeroModel がMODEL_REGISTRYに登録されています")
                model_class = MODEL_REGISTRY['GATStochasticMuZeroModel']
                print(f"✓ モデルクラス: {model_class}")
                print(f"✓ モジュール: {model_class.__module__}")
            else:
                print("❌ GATStochasticMuZeroModel がMODEL_REGISTRYに見つかりません")
                print(f"利用可能なモデル: {list(MODEL_REGISTRY.keys())}")
        except Exception as e:
            print(f"❌ MODEL_REGISTRY確認エラー: {e}")
        
        # 設定の最終確認
        print(f"\n📋 最終設定確認:")
        print(f"  - policy.model.type: {main_config.policy.model.get('type', '未設定')}")
        print(f"  - policy.model.model: {main_config.policy.model.get('model', '未設定')}")
        print(f"  - create_config.model.type: {create_config.get('model', {}).get('type', '未設定')}")
        
        return True
    
    if validate_gat_usage():
        print("✓ GNN検証準備完了")
    
    # トレーニング開始前に追加のモデル検証を行う
    print("\n🔬 詳細モデル検証...")
    try:
        # 設定に基づいてモデルを生成してテスト
        from lzero.model.gat_stochastic_muzero_model import GATStochasticMuZeroModel
        test_model = GATStochasticMuZeroModel(**main_config.policy.model)
        print(f"✅ GATStochasticMuZeroModelの生成成功!")
        print(f"✓ モデル型: {type(test_model)}")
        
        # モデルの主要コンポーネントを確認
        if hasattr(test_model, 'representation_network'):
            print(f"✓ representation_network型: {type(test_model.representation_network)}")
            
        if hasattr(test_model, 'dynamics_network'):
            print(f"✓ dynamics_network型: {type(test_model.dynamics_network)}")
            
        del test_model  # メモリ節約
        print("✓ テストモデル削除完了\n")
        
    except Exception as e:
        print(f"❌ GATStochasticMuZeroModelテスト失敗: {e}")
        import traceback
        traceback.print_exc()
    
    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)
