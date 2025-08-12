#!/usr/bin/env python3
"""
Quick training test to verify GAT is working during actual training
"""

import sys
import os
import torch

# Add LightZero to Python path
sys.path.insert(0, '/opendilab/LightZero')

print("=== GAT実トレーニング動作確認テスト ===")

# Import configuration
from zoo.game_2048.config.gat_stochastic_2048_config import main_config, create_config

# Modify config for quick test
main_config.policy.update_per_collect = 1  # Reduce for quick test
main_config.policy.batch_size = 4  # Small batch for test
main_config.policy.game_segment_length = 10  # Short segments
main_config.env.collector_env_num = 1  # Single environment
main_config.policy.num_simulations = 5  # Few simulations

print("設定を簡易テスト用に調整済み")

# Import required modules
from lzero.entry import train_muzero
# Import GAT model directly to avoid circular imports
from lzero.model.gat_stochastic_muzero_model import GATStochasticMuZeroModel

print("✓ 必要なモジュールインポート完了")

# Create a test hook to monitor GAT usage
class GATUsageMonitor:
    def __init__(self):
        self.gat_forward_calls = 0
        self.original_forward = None
    
    def hook_gat_layers(self, model):
        """Hook into GAT layers to monitor usage"""
        def make_forward_hook(layer_name):
            def forward_hook(module, input, output):
                self.gat_forward_calls += 1
                print(f"🔍 GAT Layer {layer_name} が呼び出されました！入力形状: {input[0].shape if len(input) > 0 else 'Unknown'}")
                return output
            return forward_hook
        
        # Hook representation network GAT layers
        if hasattr(model, 'representation_network') and hasattr(model.representation_network, 'gat_layers'):
            for i, layer in enumerate(model.representation_network.gat_layers):
                layer.register_forward_hook(make_forward_hook(f"Repr-{i+1}"))
                print(f"✓ Representation GAT Layer {i+1} にフック設置")
        
        # Hook afterstate dynamics GAT layers  
        if hasattr(model, 'afterstate_dynamics_network') and hasattr(model.afterstate_dynamics_network, 'gat_afterstate'):
            gat_afterstate = model.afterstate_dynamics_network.gat_afterstate
            if hasattr(gat_afterstate, 'gat_layers'):
                for i, layer in enumerate(gat_afterstate.gat_layers):
                    layer.register_forward_hook(make_forward_hook(f"Afterstate-{i+1}"))
                    print(f"✓ Afterstate GAT Layer {i+1} にフック設置")
        
        # Hook chance encoder GAT layers
        if hasattr(model, 'chance_encoder') and hasattr(model.chance_encoder, 'gat_chance'):
            gat_chance = model.chance_encoder.gat_chance
            if hasattr(gat_chance, 'gat_layers'):
                for i, layer in enumerate(gat_chance.gat_layers):
                    layer.register_forward_hook(make_forward_hook(f"Chance-{i+1}"))
                    print(f"✓ Chance GAT Layer {i+1} にフック設置")

# Test with minimal training steps
print("\n🚀 簡易トレーニング開始...")

try:
    # Set maximum environment steps to very small number for quick test
    max_env_step = 20  # Very small for quick test
    
    # Create monitor
    monitor = GATUsageMonitor()
    
    # Override model creation to add monitoring
    original_create_model = None
    
    def create_monitored_model(*args, **kwargs):
        print("🔧 モデル作成中...")
        model = GATStochasticMuZeroModel(**kwargs)
        print("✓ GATStochasticMuZeroModel作成完了")
        
        # Add monitoring hooks
        monitor.hook_gat_layers(model)
        print("✓ GATモニタリングフック設置完了")
        
        return model
    
    # Monkey patch for monitoring (this is a hack for testing)
    import lzero.model.gat_stochastic_muzero_model as gat_module
    original_model_class = gat_module.GATStochasticMuZeroModel
    
    class MonitoredGATModel(original_model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            monitor.hook_gat_layers(self)
            print("🔍 GATモニタリング有効化")
    
    # Replace the class temporarily
    gat_module.GATStochasticMuZeroModel = MonitoredGATModel
    
    print(f"最大環境ステップ数: {max_env_step}")
    print("トレーニング開始（Ctrl+Cで停止可能）...")
    
    # Run training with monitoring
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
    
    print(f"\n📊 トレーニング完了!")
    print(f"GAT Layer呼び出し回数: {monitor.gat_forward_calls}")
    
    if monitor.gat_forward_calls > 0:
        print("🎉 GAT が実際に使用されていることが確認されました！")
    else:
        print("⚠️  GAT の使用が検出されませんでした")

except KeyboardInterrupt:
    print(f"\n⏹️  ユーザーによる停止")
    print(f"GAT Layer呼び出し回数: {monitor.gat_forward_calls}")
    if monitor.gat_forward_calls > 0:
        print("🎉 GAT が実際に使用されていることが確認されました！")

except Exception as e:
    print(f"\n❌ トレーニングエラー: {e}")
    print(f"GAT Layer呼び出し回数: {monitor.gat_forward_calls}")
    if monitor.gat_forward_calls > 0:
        print("🎉 エラーは発生しましたが、GAT は実際に使用されていました！")
    import traceback
    traceback.print_exc()

print("\n=== テスト完了 ===")
