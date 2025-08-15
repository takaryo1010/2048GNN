#!/usr/bin/env python3
"""
超軽量転移学習テスト (超短時間実行版)
3×3で2分程度学習 → 4×4に1分程度転移学習のテスト
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "LightZero"))

from easydict import EasyDict


def create_ultra_light_3x3_config():
    """3×3グリッド用の超軽量テスト設定"""
    
    config = dict(
        exp_name='data_ultra_quick_transfer/phase1_3x3_ultra_quick',
        env=dict(
            stop_value=int(1e6),
            env_id='game_2048',
            obs_shape=(16, 3, 3),
            obs_type='dict_encoded_board',
            num_of_possible_chance_tile=2,
            collector_env_num=2,  # さらに軽量化（4→2）
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
                
                # GAT parameters - 軽量化
                num_heads=2,  # 4→2に削減
                hidden_channels=32,  # 64→32に削減
                num_gat_layers=2,  # 3→2に削減
                state_dim=128,  # 256→128に削減
                dropout=0.1,
                
                # StochasticMuZero parameters
                chance_encoder_num_layers=1,  # 2→1に削減
                afterstate_reward_layers=1,  # 2→1に削減
                
                # Standard parameters - 軽量化
                value_head_channels=8,  # 16→8に削減
                policy_head_channels=8,  # 16→8に削減
                value_head_hidden_channels=[16],  # [32]→[16]に削減
                policy_head_hidden_channels=[16],  # [32]→[16]に削減
                
                flatten_input_size_for_value_head=128,  # 256→128に削減
                flatten_input_size_for_policy_head=128,  # 256→128に削減
                
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
            game_segment_length=50,  # 200→50に大幅削減
            update_per_collect=10,  # 50→10に大幅削減
            batch_size=32,  # 128→32に大幅削減
            td_steps=5,  # 10→5に削減
            discount_factor=0.999,
            manual_temperature_decay=True,
            optim_type='Adam',
            piecewise_decay_lr_scheduler=False,
            learning_rate=0.005,  # 0.003→0.005に増加（高速学習）
            weight_decay=1e-4,
            num_simulations=20,  # 50→20に大幅削減
            reanalyze_ratio=0.,
            ssl_loss_weight=0,
            stochastic_loss_weight=1.0,
            n_episode=2,  # 4→2に削減
            eval_freq=int(2e3),  # 1e4→2e3に削減（頻繁に評価）
            replay_buffer_size=int(5e3),  # 1e5→5e3に大幅削減
            collector_env_num=2,
            evaluator_env_num=1,
        ),
    )
    
    return EasyDict(config)


def create_ultra_light_4x4_config(pretrained_path):
    """4×4グリッド用の超軽量転移学習設定"""
    
    config = dict(
        exp_name='data_ultra_quick_transfer/phase2_4x4_ultra_quick_transfer',
        env=dict(
            stop_value=int(1e6),
            env_id='game_2048',
            obs_shape=(16, 4, 4),
            obs_type='dict_encoded_board',
            num_of_possible_chance_tile=2,
            collector_env_num=2,
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
                observation_shape=(16, 4, 4),
                image_channel=16,
                action_space_size=4,
                chance_space_size=32,  # 4*4*2
                frame_stack_num=1,
                
                # GAT parameters (3×3と同じ軽量設定)
                num_heads=2,
                hidden_channels=32,
                num_gat_layers=2,
                state_dim=128,
                dropout=0.1,
                
                # StochasticMuZero parameters
                chance_encoder_num_layers=1,
                afterstate_reward_layers=1,
                
                # Standard parameters
                value_head_channels=8,
                policy_head_channels=8,
                value_head_hidden_channels=[16],
                policy_head_hidden_channels=[16],
                
                flatten_input_size_for_value_head=128,
                flatten_input_size_for_policy_head=128,
                
                reward_support_size=601,
                value_support_size=601,
                categorical_distribution=True,
                last_linear_layer_init_zero=True,
                state_norm=False,
                self_supervised_learning_loss=True,
            ),
            model_path=pretrained_path,  # 転移学習用
            use_ture_chance_label_in_chance_encoder=True,
            cuda=True,
            game_segment_length=50,
            update_per_collect=10,
            batch_size=32,
            td_steps=5,
            discount_factor=0.999,
            manual_temperature_decay=True,
            optim_type='Adam',
            piecewise_decay_lr_scheduler=False,
            learning_rate=0.002,  # 転移学習では0.005→0.002に削減
            weight_decay=1e-4,
            num_simulations=20,
            reanalyze_ratio=0.,
            ssl_loss_weight=0,
            stochastic_loss_weight=1.0,
            n_episode=2,
            eval_freq=int(2e3),
            replay_buffer_size=int(5e3),
            collector_env_num=2,
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
    """超軽量転移学習テスト"""
    
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
    
    print("=" * 70)
    print("⚡ GAT転移学習 超軽量テスト")
    print("🚀 超短時間実行版 (3×3 → 4×4)")
    print("=" * 70)
    print("📊 軽量化設定:")
    print("  - 環境数: 2 (通常8)")
    print("  - バッチサイズ: 32 (通常512)")
    print("  - シミュレーション: 20 (通常100)")
    print("  - GAT層: 2 (通常3)")
    print("  - 隠れ次元: 128 (通常256)")
    print("  - アップデート頻度: 10 (通常200)")
    print("=" * 70)
    
    # Phase 1: 3×3超軽量学習
    print("\n🔵 Phase 1: 3×3超軽量学習")
    config_3x3 = create_ultra_light_3x3_config()
    create_config = get_create_config()
    
    print(f"✓ 実験名: {config_3x3.exp_name}")
    print(f"✓ チャンス空間: {config_3x3.policy.model.chance_space_size}")
    print(f"✓ 超軽量設定: バッチ{config_3x3.policy.batch_size}, シミュ{config_3x3.policy.num_simulations}")
    print(f"✓ 目標ステップ: {int(5e3)} (非常に短時間)")
    
    try:
        print("🚀 3×3超軽量学習開始...")
        train_muzero([config_3x3, create_config], seed=0, model_path=None, max_env_step=int(5e3))  # 非常に短時間
        print("✅ 3×3学習完了")
    except Exception as e:
        logger.error(f"❌ 3×3学習エラー: {e}")
        return False
    
    # チェックポイント検索
    import glob
    checkpoint_pattern = "./data_ultra_quick_transfer/phase1_3x3_**/ckpt_*.pth.tar"
    checkpoints = glob.glob(checkpoint_pattern, recursive=True)
    
    if not checkpoints:
        logger.error("❌ チェックポイントが見つかりません")
        return False
    
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"✅ チェックポイント発見: {latest_checkpoint}")
    
    # Phase 2: 4×4転移学習
    print("\n🔴 Phase 2: 4×4超軽量転移学習")
    config_4x4 = create_ultra_light_4x4_config(latest_checkpoint)
    
    print(f"✓ 実験名: {config_4x4.exp_name}")
    print(f"✓ チャンス空間: {config_4x4.policy.model.chance_space_size}")
    print(f"✓ プリトレイン: {latest_checkpoint}")
    print(f"✓ 目標ステップ: {int(3e3)} (短時間)")
    
    try:
        print("🚀 4×4転移学習開始...")
        train_muzero([config_4x4, create_config], seed=0, model_path=latest_checkpoint, max_env_step=int(3e3))
        print("✅ 4×4転移学習完了")
    except Exception as e:
        logger.error(f"❌ 4×4転移学習エラー: {e}")
        return False
    
    print("\n🎉 超軽量転移学習テスト完了！")
    print("⏱️  総実行時間: 約3-5分を想定")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
