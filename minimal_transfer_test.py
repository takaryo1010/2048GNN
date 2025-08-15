#!/usr/bin/env python3
"""
最小検証テスト (1分程度で完了)
転移学習の動作確認のみを行う
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

from easydict import EasyDict


def create_minimal_3x3_config():
    """3×3グリッド用の最小テスト設定"""
    
    config = dict(
        exp_name='data_minimal_transfer/phase1_3x3_minimal',
        env=dict(
            stop_value=int(1e6),
            env_id='game_2048',
            obs_shape=(16, 3, 3),
            obs_type='dict_encoded_board',
            num_of_possible_chance_tile=2,
            collector_env_num=1,  # 最小限に削減
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
                observation_shape=(16, 3, 3),
                image_channel=16,
                action_space_size=4,
                chance_space_size=18,  # 3*3*2
                frame_stack_num=1,
                
                # GAT parameters - 最小構成
                num_heads=1,  # 最小限
                hidden_channels=16,  # 最小限
                num_gat_layers=1,  # 最小限
                state_dim=64,  # 最小限
                dropout=0.0,  # 無効化
                
                # StochasticMuZero parameters - 最小構成
                chance_encoder_num_layers=1,
                afterstate_reward_layers=1,
                
                # Standard parameters - 最小構成
                value_head_channels=4,
                policy_head_channels=4,
                value_head_hidden_channels=[8],
                policy_head_hidden_channels=[8],
                
                flatten_input_size_for_value_head=64,
                flatten_input_size_for_policy_head=64,
                
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
            game_segment_length=20,  # 最小限
            update_per_collect=5,  # 最小限
            batch_size=8,  # 最小限
            td_steps=3,  # 最小限
            discount_factor=0.999,
            manual_temperature_decay=True,
            optim_type='Adam',
            piecewise_decay_lr_scheduler=False,
            learning_rate=0.01,  # 高速学習
            weight_decay=1e-4,
            num_simulations=5,  # 最小限
            reanalyze_ratio=0.,
            ssl_loss_weight=0,
            stochastic_loss_weight=1.0,
            n_episode=1,  # 最小限
            eval_freq=int(5e2),  # 頻繁に評価
            replay_buffer_size=int(1e3),  # 最小限
            collector_env_num=1,
            evaluator_env_num=1,
        ),
    )
    
    return EasyDict(config)


def create_minimal_4x4_config(checkpoint_path=None):
    """4×4グリッド用の最小設定"""
    config = dict(
        exp_name='phase2_4x4_minimal',
        env=dict(
            grid_size=4,
            max_episode_steps=100,  # 短縮
        ),
        policy=dict(
            type='stochastic_muzero',
            model=dict(
                type='GATStochasticMuZeroModel',
                model_type='gat',
                model='gat_stochastic',
                observation_shape=(16, 4, 4),
                image_channel=16,
                action_space_size=4,
                chance_space_size=32,  # 4*4*2
                frame_stack_num=1,
                
                # GAT parameters (3×3と同じ最小設定)
                num_heads=1,
                hidden_channels=16,
                num_gat_layers=1,
                state_dim=64,
                dropout=0.0,
                
                # StochasticMuZero parameters
                chance_encoder_num_layers=1,
                dynamics_num_layers=1,
                prediction_num_layers=1,
                categorical_distribution=True,
                image_transform=True,
                self_supervised_learning_loss=False,
                consistency_loss=False,
                supports=list(range(-300, 301)),
            ),
            # チェックポイントロード設定
            load_path=checkpoint_path,
        ),
        train=dict(
            work_dir='./data_minimal_transfer/phase2_4x4_minimal',
            stop_value=1000000,
            update_per_collect=5,  # 最小限
            batch_size=8,  # 最小限
            td_steps=3,  # 最小限
            discount_factor=0.999,
            manual_temperature_decay=True,
            optim_type='Adam',
            piecewise_decay_lr_scheduler=False,
            learning_rate=0.01,  # 高速学習
            weight_decay=1e-4,
            num_simulations=5,  # 最小限
            reanalyze_ratio=0.,
            ssl_loss_weight=0,
            stochastic_loss_weight=1.0,
            n_episode=1,  # 最小限
            eval_freq=int(5e2),  # 頻繁に評価
            replay_buffer_size=int(1e3),  # 最小限
            collector_env_num=1,
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
    """最小転移学習テスト"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # インポート
    try:
        from lzero.entry import train_muzero
        from lzero.model.gat_stochastic_muzero_model import GATStochasticMuZeroModel
        logger.info("✅ LightZero インポート成功")
    except ImportError as e:
        logger.error(f"❌ インポートエラー: {e}")
        return False
    
    print("=" * 60)
    print("⚡ GAT転移学習 最小検証テスト")
    print("🏃 1分程度で完了予定")
    print("=" * 60)
    print("📊 最小構成:")
    print("  - 環境数: 1")
    print("  - バッチサイズ: 8")
    print("  - シミュレーション: 5")
    print("  - GAT層: 1")
    print("  - 隠れ次元: 64")
    print("=" * 60)
    
    # Phase 1: 3×3最小学習
    print("\n🔵 Phase 1: 3×3最小学習 (1000ステップ)")
    config_3x3 = create_minimal_3x3_config()
    create_config = get_create_config()
    
    try:
        print("🚀 3×3最小学習開始...")
        train_muzero([config_3x3, create_config], seed=0, model_path=None, max_env_step=int(1e3))
        print("✅ 3×3学習完了")
    except Exception as e:
        logger.error(f"❌ 3×3学習エラー: {e}")
        return False
    
    # チェックポイント検索
    import glob
    checkpoint_pattern = "./data_minimal_transfer/phase1_3x3_minimal/ckpt/*.pth.tar"
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        logger.error("❌ チェックポイントが見つかりません")
        return False
    
    # ベストチェックポイントを優先、なければ最新を使用
    best_checkpoint = "./data_minimal_transfer/phase1_3x3_minimal/ckpt/ckpt_best.pth.tar"
    if os.path.exists(best_checkpoint):
        latest_checkpoint = best_checkpoint
    else:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"✅ チェックポイント発見: {latest_checkpoint}")
    
    # Phase 2: 4×4転移学習
    print("\n🔴 Phase 2: 4×4最小転移学習 (500ステップ)")
    config_4x4 = create_minimal_4x4_config(latest_checkpoint)
    
    try:
        print("🚀 4×4転移学習開始...")
        train_muzero([config_4x4, create_config], seed=0, model_path=latest_checkpoint, max_env_step=int(5e2))
        print("✅ 4×4転移学習完了")
    except Exception as e:
        logger.error(f"❌ 4×4転移学習エラー: {e}")
        return False
    
    print("\n🎉 最小転移学習検証完了！")
    print("✅ 転移学習の基本動作が確認されました")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
