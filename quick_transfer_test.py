#!/usr/bin/env python3
"""
簡易転移学習テスト (短時間実行版)
3×3で短時間学習 → 4×4に転移学習のテスト
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


def create_quick_3x3_config():
    """3×3グリッド用の短時間テスト設定"""
    
    config = dict(
        exp_name='data_quick_transfer/phase1_3x3_quick_test',
        env=dict(
            stop_value=int(1e6),
            env_id='game_2048',
            obs_shape=(16, 3, 3),
            obs_type='dict_encoded_board',
            num_of_possible_chance_tile=2,
            collector_env_num=2,  # さらに軽量化
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
                
                # GAT parameters
                num_heads=4,
                hidden_channels=64,
                num_gat_layers=3,
                state_dim=256,
                dropout=0.1,
                
                # StochasticMuZero parameters
                chance_encoder_num_layers=2,
                afterstate_reward_layers=2,
                
                # Standard parameters
                value_head_channels=16,
                policy_head_channels=16,
                value_head_hidden_channels=[32],
                policy_head_hidden_channels=[32],
                
                flatten_input_size_for_value_head=256,
                flatten_input_size_for_policy_head=256,
                
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
            update_per_collect=50,  # 軽量化
            batch_size=64,  # 軽量化（さらに削減）
            td_steps=10,
            discount_factor=0.999,
            manual_temperature_decay=True,
            optim_type='Adam',
            piecewise_decay_lr_scheduler=False,
            learning_rate=0.003,
            weight_decay=1e-4,
            num_simulations=25,  # 軽量化（さらに削減）
            reanalyze_ratio=0.,
            ssl_loss_weight=0,
            stochastic_loss_weight=1.0,
            n_episode=8,  # collector_env_num(2) * 4
            eval_freq=int(5e3),  # より頻繁に評価
            replay_buffer_size=int(5e4),  # 軽量化（さらに削減）
            collector_env_num=2,
            evaluator_env_num=1,
        ),
    )
    
    return EasyDict(config)


def create_quick_4x4_config(pretrained_path):
    """4×4グリッド用の転移学習設定"""
    
    config = dict(
        exp_name='data_quick_transfer/phase2_4x4_transfer_test',
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
                
                # GAT parameters (同じ設定)
                num_heads=4,
                hidden_channels=64,
                num_gat_layers=3,
                state_dim=256,
                dropout=0.1,
                
                # StochasticMuZero parameters
                chance_encoder_num_layers=2,
                afterstate_reward_layers=2,
                
                # Standard parameters
                value_head_channels=16,
                policy_head_channels=16,
                value_head_hidden_channels=[32],
                policy_head_hidden_channels=[32],
                
                flatten_input_size_for_value_head=256,
                flatten_input_size_for_policy_head=256,
                
                reward_support_size=601,
                value_support_size=601,
                categorical_distribution=True,
                last_linear_layer_init_zero=True,
                state_norm=False,
                self_supervised_learning_loss=True,
            ),
            model_path=None,  # 後で安全転移学習を実行
            use_ture_chance_label_in_chance_encoder=True,
            cuda=True,
            game_segment_length=200,
            update_per_collect=50,
            batch_size=64,
            td_steps=10,
            discount_factor=0.999,
            manual_temperature_decay=True,
            optim_type='Adam',
            piecewise_decay_lr_scheduler=False,
            learning_rate=0.001,  # 転移学習では低い学習率
            weight_decay=1e-4,
            num_simulations=25,
            reanalyze_ratio=0.,
            ssl_loss_weight=0,
            stochastic_loss_weight=1.0,
            n_episode=8,
            eval_freq=int(5e3),
            replay_buffer_size=int(5e4),
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


def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description='GAT転移学習テスト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 通常の実行（3×3学習 → 4×4転移学習）
  python quick_transfer_test.py
  
  # 既存の3×3モデルを使用して4×4転移学習のみ実行
  python quick_transfer_test.py --skip-3x3 --model-path ./data_quick_transfer/phase1_3x3_xxx/ckpt/iteration_400.pth.tar
  
  # ステップ数を調整
  python quick_transfer_test.py --max-steps-3x3 10000 --max-steps-4x4 5000
        """
    )
    parser.add_argument(
        '--skip-3x3', 
        action='store_true', 
        help='3×3学習をスキップして既存モデルを使用'
    )
    parser.add_argument(
        '--model-path', 
        type=str, 
        help='既存の3×3モデルのパス (--skip-3x3使用時に必要)'
    )
    parser.add_argument(
        '--max-steps-3x3',
        type=int,
        default=5000,
        help='3×3学習の最大ステップ数 (デフォルト: 5000)'
    )
    parser.add_argument(
        '--max-steps-4x4',
        type=int,
        default=3000,
        help='4×4転移学習の最大ステップ数 (デフォルト: 3000)'
    )
    return parser.parse_args()


def main():
    """簡易転移学習テスト"""
    
    args = parse_arguments()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # インポート
    try:
        from lzero.entry import train_muzero
        from lzero.model.gat_stochastic_muzero_model import GATStochasticMuZeroModel
        from gat_transfer_helper import safe_load_transfer_weights
        logger.info("✅ LightZero インポート成功")
    except ImportError as e:
        logger.error(f"❌ インポートエラー: {e}")
        return False
    
    print("=" * 70)
    print("🧪 GAT転移学習 簡易テスト")
    print("⏱️  短時間実行版 (3×3 → 4×4)")
    print("=" * 70)
    
    # 共通設定
    create_config = get_create_config()
    
    latest_checkpoint = None
    
    # Phase 1: 3×3学習（スキップ可能）
    if args.skip_3x3:
        if not args.model_path:
            logger.error("❌ --skip-3x3を使用する場合は--model-pathが必要です")
            return False
        
        if not os.path.exists(args.model_path):
            logger.error(f"❌ 指定されたモデルパスが存在しません: {args.model_path}")
            return False
        
        latest_checkpoint = args.model_path
        print(f"\n⏭️  3×3学習をスキップ")
        print(f"✓ 既存モデル使用: {latest_checkpoint}")
        
    else:
        print("\n🔵 Phase 1: 3×3短時間学習")
        config_3x3 = create_quick_3x3_config()
        
        print(f"✓ 実験名: {config_3x3.exp_name}")
        print(f"✓ チャンス空間: {config_3x3.policy.model.chance_space_size}")
        print(f"✓ 軽量設定: バッチ{config_3x3.policy.batch_size}, シミュ{config_3x3.policy.num_simulations}")
        print(f"✓ 最大ステップ: {args.max_steps_3x3}")
        
        try:
            print("🚀 3×3学習開始...")
            train_muzero([config_3x3, create_config], seed=0, model_path=None, max_env_step=args.max_steps_3x3)
            print("✅ 3×3学習完了")
        except Exception as e:
            logger.error(f"❌ 3×3学習エラー: {e}")
            return False
        
        # チェックポイント検索
        import glob
        
        # 複数のパターンを試す
        checkpoint_patterns = [
            "./data_quick_transfer/phase1_3x3_**/ckpt/*.pth.tar",
            "./data_quick_transfer/phase1_3x3_**/ckpt_*.pth.tar",
            "./data_quick_transfer/phase1_3x3_**/*.pth.tar",
            "./data_quick_transfer/**/ckpt/*.pth.tar"
        ]
        
        checkpoints = []
        for pattern in checkpoint_patterns:
            found = glob.glob(pattern, recursive=True)
            checkpoints.extend(found)
            if found:
                print(f"✓ パターン '{pattern}' で発見: {len(found)}個")
        
        if not checkpoints:
            # ディレクトリを直接確認
            print("📁 利用可能なディレクトリを確認中...")
            for exp_dir in Path("./data_quick_transfer").glob("phase1_3x3_*"):
                print(f"  - {exp_dir}")
                for ckpt_file in exp_dir.rglob("*.pth.tar"):
                    print(f"    ✓ {ckpt_file}")
                    checkpoints.append(str(ckpt_file))
        
        if not checkpoints:
            logger.error("❌ チェックポイントが見つかりません")
            return False
        
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"✅ チェックポイント発見: {latest_checkpoint}")
    
    # Phase 2: 4×4転移学習
    print("\n🔴 Phase 2: 4×4転移学習")
    config_4x4 = create_quick_4x4_config(latest_checkpoint)
    
    print(f"✓ 実験名: {config_4x4.exp_name}")
    print(f"✓ チャンス空間: {config_4x4.policy.model.chance_space_size}")
    print(f"✓ プリトレイン: {latest_checkpoint}")
    print(f"✓ 最大ステップ: {args.max_steps_4x4}")
    
    try:
        print("🚀 4×4モデル初期化...")
        # 4×4モデルを作成して安全転移学習
        from lzero.model.gat_stochastic_muzero_model import GATStochasticMuZeroModel
        
        model_4x4 = GATStochasticMuZeroModel(
            observation_shape=(16, 4, 4),
            action_space_size=4,
            chance_space_size=32,
            num_heads=4,
            hidden_channels=64,
            num_gat_layers=3,
            state_dim=256,
            dropout=0.1
        )
        
        print("🔄 安全転移学習実行...")
        safe_load_transfer_weights(model_4x4, latest_checkpoint)
        
        print("🚀 4×4転移学習開始...")
        train_muzero([config_4x4, create_config], seed=0, model_path=None, max_env_step=args.max_steps_4x4)
        print("✅ 4×4転移学習完了")
    except Exception as e:
        logger.error(f"❌ 4×4転移学習エラー: {e}")
        return False
    
    print("\n🎉 簡易転移学習テスト完了！")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
