#!/usr/bin/env python3
"""
Test script to verify that GAT is actually being used in the model
"""

import sys
import os
import torch
import torch.nn as nn

# Add LightZero to Python path
sys.path.insert(0, '/opendilab/LightZero')

print("=== GAT 使用状況確認テスト ===")

# Test 1: Import the GAT model directly
print("\n1. GATモデルの直接インポートテスト...")
try:
    from lzero.model.gat_stochastic_muzero_model import GATStochasticMuZeroModel
    print("✓ GATStochasticMuZeroModelの正常インポート")
except Exception as e:
    print(f"✗ GATStochasticMuZeroModelのインポートエラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Create a simple model instance to check GAT components
print("\n2. GATモデルインスタンスの作成と構成確認...")
try:
    model_config = {
        'observation_shape': (16, 4, 4),
        'action_space_size': 4,
        'chance_space_size': 32,
        'num_heads': 4,
        'hidden_channels': 64,
        'num_gat_layers': 3,
        'state_dim': 256,
        'dropout': 0.1,
        'grid_size': 4,
        'chance_encoder_num_layers': 2,
        'afterstate_reward_layers': 2,
        'value_support_size': 601,
        'reward_support_size': 601,
        'value_head_channels': 16,
        'policy_head_channels': 16,
        'categorical_distribution': True
    }
    
    model = GATStochasticMuZeroModel(**model_config)
    print("✓ GATStochasticMuZeroModelインスタンス作成成功")
    
    # Check for GAT components
    print("\n3. GATコンポーネントの存在確認...")
    
    # Check representation network
    if hasattr(model, 'representation_network'):
        repr_net = model.representation_network
        print(f"✓ representation_network存在: {type(repr_net)}")
        
        # Check for GAT layers
        if hasattr(repr_net, 'gat_layers'):
            gat_layers = repr_net.gat_layers
            print(f"✓ GAT layers存在: {len(gat_layers)}層")
            
            # Check each GAT layer
            for i, layer in enumerate(gat_layers):
                if hasattr(layer, 'heads'):
                    print(f"  - GAT Layer {i+1}: {layer.heads}ヘッド, in_channels={layer.in_channels}, out_channels={layer.out_channels}")
                else:
                    print(f"  - Layer {i+1}: {type(layer)} (GAT以外)")
        else:
            print("✗ GAT layersが見つかりません")
    else:
        print("✗ representation_networkが見つかりません")
    
    # Check if grid to graph converter exists
    if hasattr(model.representation_network, 'converter'):
        converter = model.representation_network.converter
        print(f"✓ GridToGraphConverter存在: グリッドサイズ {converter.grid_size}x{converter.grid_size}")
        print(f"  - ノード数: {converter.num_nodes}")
        print(f"  - エッジ数: {converter.edge_index.shape[1] if converter.edge_index is not None else 'Unknown'}")
    else:
        print("✗ GridToGraphConverterが見つかりません")
        
except Exception as e:
    print(f"✗ モデル作成エラー: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Forward pass with dummy data to see if GAT is actually processing
print("\n4. ダミーデータでの前向き処理テスト...")
try:
    # Create dummy observation
    batch_size = 2
    dummy_obs = torch.randn(batch_size, 16, 4, 4)  # (batch, channels, height, width)
    
    print(f"ダミー観測データ形状: {dummy_obs.shape}")
    
    # Run representation network
    with torch.no_grad():
        encoded_state = model.representation_network(dummy_obs)
        print(f"✓ 前向き処理成功")
        print(f"エンコード状態形状: {encoded_state.shape}")
        
        # Check if the encoded state has the expected dimensions for GAT output
        expected_dim = model_config['state_dim']
        if encoded_state.shape[-1] == expected_dim:
            print(f"✓ 出力次元が期待値と一致: {expected_dim}")
        else:
            print(f"⚠ 出力次元が期待値と異なる: 期待値={expected_dim}, 実際={encoded_state.shape[-1]}")

except Exception as e:
    print(f"✗ 前向き処理エラー: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check torch_geometric dependency
print("\n5. PyTorch Geometric依存関係の確認...")
try:
    import torch_geometric
    from torch_geometric.nn import GATConv
    print(f"✓ PyTorch Geometric バージョン: {torch_geometric.__version__}")
    print("✓ GATConv正常インポート")
except Exception as e:
    print(f"✗ PyTorch Geometric インポートエラー: {e}")

print("\n=== テスト完了 ===")
print("\n結論:")
print("- GATモデルが定義されており、GAT layersが含まれている場合、GATが使用されています")
print("- 前向き処理が成功している場合、実際にGATによる処理が行われています")
print("- GridToGraphConverterがグリッドデータをグラフ形式に変換しています")
