#!/usr/bin/env python3
"""
GAT StochasticMuZero 段階的転移学習スクリプト
3×3で学習 → 4×4に転移学習
"""

import os
import sys
import torch
import logging
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "LightZero"))

from transfer_learning_helper import TransferLearningHelper, create_transfer_config
from easydict import EasyDict


def create_3x3_config():
    """3×3グリッド用の設定を作成"""
    
    # 基本設定
    env_id = 'game_2048'
    grid_size = 3
    action_space_size = 4
    num_of_possible_chance_tile = 2
    chance_space_size = (grid_size ** 2) * num_of_possible_chance_tile  # 18
    
    # GAT設定
    num_heads = 4
    hidden_channels = 64
    num_gat_layers = 3
    state_dim = 256
    dropout = 0.1
    
    # トレーニング設定
    collector_env_num = 8
    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 100
    update_per_collect = 200
    batch_size = 512
    reanalyze_ratio = 0.
    
    config = dict(
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
                chance_encoder_num_layers=2,
                afterstate_reward_layers=2,
                
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
            use_ture_chance_label_in_chance_encoder=True,
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
            stochastic_loss_weight=1.0,
            n_episode=n_episode,
            eval_freq=int(2e5),  # 転移のために短めに設定
            replay_buffer_size=int(1e6),
            collector_env_num=collector_env_num,
            evaluator_env_num=1,
        ),
    )
    
    return EasyDict(config)


def create_4x4_config():
    """4×4グリッド用の設定を作成（転移学習前のベース）"""
    
    # 基本設定
    env_id = 'game_2048'
    grid_size = 4
    action_space_size = 4
    num_of_possible_chance_tile = 2
    chance_space_size = (grid_size ** 2) * num_of_possible_chance_tile  # 32
    
    # GAT設定（3×3と同じ）
    num_heads = 4
    hidden_channels = 64
    num_gat_layers = 3
    state_dim = 256
    dropout = 0.1
    
    # トレーニング設定
    collector_env_num = 8
    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 100
    update_per_collect = 200
    batch_size = 512
    reanalyze_ratio = 0.
    
    config = dict(
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
                
                # GAT parameters
                num_heads=num_heads,
                hidden_channels=hidden_channels,
                num_gat_layers=num_gat_layers,
                state_dim=state_dim,
                dropout=dropout,
                
                # StochasticMuZero parameters
                chance_encoder_num_layers=2,
                afterstate_reward_layers=2,
                
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
            model_path=None,  # 転移学習時に設定される
            use_ture_chance_label_in_chance_encoder=True,
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
            stochastic_loss_weight=1.0,
            n_episode=n_episode,
            eval_freq=int(1e6),
            replay_buffer_size=int(1e6),
            collector_env_num=collector_env_num,
            evaluator_env_num=1,
        ),
    )
    
    return EasyDict(config)


def get_create_config():
    """共通のcreate_config"""
    return EasyDict(dict(
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
    ))


def main():
    """メイン転移学習プロセス"""
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # インポート
    try:
        from lzero.entry import train_muzero
        from lzero.model.gat_stochastic_muzero_model import GATStochasticMuZeroModel
        logger.info("✅ LightZero モジュールのインポート成功")
    except ImportError as e:
        logger.error(f"❌ LightZero モジュールのインポートエラー: {e}")
        return False
    
    # 転移学習ヘルパー
    helper = TransferLearningHelper()
    
    print("=" * 80)
    print("🚀 GAT STOCHASTIC MUZERO 段階的転移学習")
    print("📈 3×3グリッド → 4×4グリッド")
    print("=" * 80)
    
    # ==========================================================================
    # PHASE 1: 3×3グリッドでの学習
    # ==========================================================================
    print("\n🔵 PHASE 1: 3×3グリッドでの基礎学習")
    print("=" * 60)
    
    config_3x3 = create_3x3_config()
    create_config = get_create_config()
    
    print(f"✓ 3×3設定:")
    print(f"  - 実験名: {config_3x3.exp_name}")
    print(f"  - チャンス空間: {config_3x3.policy.model.chance_space_size}")
    print(f"  - GAT設定: {config_3x3.policy.model.num_heads}ヘッド, {config_3x3.policy.model.num_gat_layers}層")
    print(f"  - 学習率: {config_3x3.policy.learning_rate}")
    
    # 3×3での学習実行
    print("\n🚀 3×3グリッドでのトレーニング開始...")
    try:
        max_env_step_phase1 = int(3e5)  # 転移のために短めに設定
        train_muzero([config_3x3, create_config], seed=0, model_path=None, max_env_step=max_env_step_phase1)
        print("✅ Phase 1 (3×3) 学習完了")
    except Exception as e:
        logger.error(f"❌ Phase 1 学習エラー: {e}")
        return False
    
    # 学習済みモデルの検索
    print("\n🔍 Phase 1 学習済みモデルの検索...")
    phase1_checkpoint = helper.find_latest_checkpoint("./data_gat_transfer/phase1_3x3_*")
    
    if not phase1_checkpoint:
        logger.error("❌ Phase 1のチェックポイントが見つかりません")
        return False
    
    print(f"✅ Phase 1チェックポイント発見: {phase1_checkpoint}")
    
    # ==========================================================================
    # PHASE 2: 4×4グリッドへの転移学習
    # ==========================================================================
    print("\n🔴 PHASE 2: 4×4グリッドへの転移学習")
    print("=" * 60)
    
    config_4x4_base = create_4x4_config()
    
    # 転移学習設定の作成
    config_4x4_transfer = create_transfer_config(
        config_3x3, 
        target_grid_size=4, 
        pretrained_model_path=phase1_checkpoint
    )
    
    # 互換性確認
    is_compatible, compatibility_msg = helper.validate_transfer_compatibility(config_3x3, config_4x4_base)
    print(f"🔍 転移学習互換性確認: {compatibility_msg}")
    
    if not is_compatible:
        logger.error("❌ 転移学習互換性エラー")
        return False
    
    print(f"✓ 4×4転移設定:")
    print(f"  - 実験名: {config_4x4_transfer.exp_name}")
    print(f"  - チャンス空間: {config_4x4_transfer.policy.model.chance_space_size}")
    print(f"  - プリトレインモデル: {config_4x4_transfer.policy.model_path}")
    print(f"  - 転移学習率: {config_4x4_transfer.policy.learning_rate}")
    
    # 4×4での転移学習実行
    print("\n🚀 4×4グリッドでの転移学習開始...")
    try:
        max_env_step_phase2 = int(1e6)
        train_muzero([config_4x4_transfer, create_config], seed=0, 
                    model_path=phase1_checkpoint, max_env_step=max_env_step_phase2)
        print("✅ Phase 2 (4×4転移学習) 完了")
    except Exception as e:
        logger.error(f"❌ Phase 2 転移学習エラー: {e}")
        return False
    
    # ==========================================================================
    # 完了
    # ==========================================================================
    print("\n🎉 転移学習プロセス完了！")
    print("=" * 80)
    print("📊 結果サマリー:")
    print(f"  ✓ Phase 1 (3×3): 完了")
    print(f"  ✓ Phase 2 (4×4転移): 完了")
    print(f"  📁 Phase 1 結果: ./data_gat_transfer/phase1_3x3_*")
    print(f"  📁 Phase 2 結果: ./data_gat_transfer/phase2_4x4_*")
    
    # 最終チェックポイント検索
    phase2_checkpoint = helper.find_latest_checkpoint("./data_gat_transfer/phase2_4x4_*")
    if phase2_checkpoint:
        print(f"  📥 最終モデル: {phase2_checkpoint}")
    
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
