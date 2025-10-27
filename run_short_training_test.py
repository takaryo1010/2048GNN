"""
実際のトレーニングを短時間実行してGNNが学習されることを確認
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'LightZero'))

from zoo.game_2048.config.stochastic_muzero_2048_gnn_config import main_config, create_config
from lzero.entry import train_muzero

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 GNN実トレーニングテスト")
    print("="*80)
    print("\n📋 設定情報:")
    print(f"   - Model Type: {create_config.model.type}")
    print(f"   - GNN Layers: {main_config.policy.model.num_gnn_layers}")
    print(f"   - GNN Channels: {main_config.policy.model.num_channels}")
    print(f"   - Edge Mode: {main_config.policy.model.edge_mode}")
    print(f"   - Batch Size: {main_config.policy.batch_size}")
    print(f"   - Update per Collect: {main_config.policy.update_per_collect}")
    
    # 短時間トレーニング用の設定
    from copy import deepcopy
    test_config = deepcopy(main_config)
    test_config['policy']['eval_freq'] = int(200)  # 200環境ステップごとに評価
    
    max_env_step = int(2000)  # 2000環境ステップで終了
    
    print(f"\n🎯 テスト設定:")
    print(f"   - 最大環境ステップ: {max_env_step}")
    print(f"   - 評価頻度: {test_config['policy']['eval_freq']}")
    print(f"   - 予想時間: 約5-10分")
    
    print("\n" + "="*80)
    print("🏋️  トレーニング開始...")
    print("="*80 + "\n")
    
    try:
        train_muzero(
            [test_config, create_config],
            seed=0,
            max_env_step=max_env_step,
        )
        print("\n✅ トレーニングテスト完了！")
    except KeyboardInterrupt:
        print("\n⚠️  ユーザーによって中断されました")
    except Exception as e:
        print(f"\n❌ エラーが発生: {e}")
        import traceback
        traceback.print_exc()
