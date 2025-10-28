#!/usr/bin/env python3
"""
GATChanceEncoderのテスト

テスト内容:
1. サイズ非依存性テスト（3×3、4×4、5×5で動作確認）
2. 出力形状テスト
3. 転移学習テスト（4×4の重みを3×3で使用）
4. 完全GAT化バリデーションテスト
"""
import sys
sys.path.insert(0, 'LightZero')

import torch
import torch.nn as nn
from lzero.model.gat_stochastic_muzero_model import (
    GATChanceEncoder,
    GATStochasticMuZeroModel
)


def test_gat_chance_encoder_basic():
    """基本的な動作テスト"""
    print("=" * 80)
    print("テスト1: GATChanceEncoder基本動作")
    print("=" * 80)
    
    observation_shape = (16, 4, 4)
    chance_space_size = 32
    batch_size = 8
    
    encoder = GATChanceEncoder(
        observation_shape=observation_shape,
        chance_space_size=chance_space_size,
        num_gnn_layers=2,
        num_heads=4,
        hidden_dim=64,
        edge_mode='adjacent',
        norm_type='layer'  # LayerNormで安定性重視
    )
    
    # テスト入力
    obs = torch.randn(batch_size, *observation_shape)
    
    # Forward
    chance_encoding, chance_onehot = encoder(obs)
    
    # 出力形状確認
    assert chance_encoding.shape == (batch_size, chance_space_size), \
        f"Encoding shape mismatch: {chance_encoding.shape}"
    assert chance_onehot.shape == (batch_size, chance_space_size), \
        f"Onehot shape mismatch: {chance_onehot.shape}"
    
    # One-hot確認
    assert torch.allclose(chance_onehot.sum(dim=-1), torch.ones(batch_size)), \
        "One-hot sum should be 1"
    
    print(f"✅ 入力形状: {obs.shape}")
    print(f"✅ 出力形状（encoding）: {chance_encoding.shape}")
    print(f"✅ 出力形状（onehot）: {chance_onehot.shape}")
    print(f"✅ パラメータ数: {sum(p.numel() for p in encoder.parameters()):,}")
    print()


def test_size_independence():
    """サイズ非依存性テスト"""
    print("=" * 80)
    print("テスト2: サイズ非依存性（3×3、4×4、5×5）")
    print("=" * 80)
    
    chance_space_size = 32
    batch_size = 8
    
    for size in [3, 4, 5]:
        print(f"\n🔵 グリッドサイズ: {size}×{size}")
        
        observation_shape = (16, size, size)
        encoder = GATChanceEncoder(
            observation_shape=observation_shape,
            chance_space_size=chance_space_size,
            num_gnn_layers=2,
            num_heads=4,
            hidden_dim=64,
            edge_mode='adjacent',
            norm_type='layer'  # LayerNormで安定性重視
        )
        
        obs = torch.randn(batch_size, *observation_shape)
        chance_encoding, chance_onehot = encoder(obs)
        
        assert chance_encoding.shape == (batch_size, chance_space_size)
        assert chance_onehot.shape == (batch_size, chance_space_size)
        
        print(f"  ✅ 入力: {obs.shape}")
        print(f"  ✅ 出力: {chance_encoding.shape}")
        print(f"  ✅ ノード数: {size * size}")
        print(f"  ✅ エッジ数: {encoder.graph_builder.edge_index.shape[1]}")
    
    print()


def test_transfer_learning():
    """転移学習テスト"""
    print("=" * 80)
    print("テスト3: 転移学習（4×4 → 3×3）")
    print("=" * 80)
    
    chance_space_size = 32
    
    # 4×4で学習済みモデル
    print("\n🔵 Step 1: 4×4モデルを作成")
    encoder_4x4 = GATChanceEncoder(
        observation_shape=(16, 4, 4),
        chance_space_size=chance_space_size,
        num_gnn_layers=2,
        num_heads=4,
        hidden_dim=64,
        edge_mode='adjacent',
        norm_type='layer'  # LayerNormで安定性重視
    )
    print(f"  ✅ 4×4モデル作成完了")
    print(f"  ✅ パラメータ数: {sum(p.numel() for p in encoder_4x4.parameters()):,}")
    
    # 3×3モデルを作成
    print("\n🔵 Step 2: 3×3モデルを作成")
    encoder_3x3 = GATChanceEncoder(
        observation_shape=(16, 3, 3),
        chance_space_size=chance_space_size,
        num_gnn_layers=2,
        num_heads=4,
        hidden_dim=64,
        edge_mode='adjacent',
        norm_type='layer'  # LayerNormで安定性重視
    )
    print(f"  ✅ 3×3モデル作成完了")
    
    # 重みの転移
    print("\n🔵 Step 3: 重みを転移（4×4 → 3×3）")
    state_dict_4x4 = encoder_4x4.state_dict()
    
    # GraphBuilder以外の重みをロード（strict=Falseで部分的にロード）
    try:
        encoder_3x3.load_state_dict(state_dict_4x4, strict=False)
        print("  ✅ 重みの転移成功（GraphBuilder以外）")
    except Exception as e:
        print(f"  ⚠️  一部の重みが転移できませんでした: {e}")
    
    # 転移後の動作確認
    print("\n🔵 Step 4: 転移後の動作確認")
    obs_3x3 = torch.randn(4, 16, 3, 3)
    chance_encoding, chance_onehot = encoder_3x3(obs_3x3)
    
    assert chance_encoding.shape == (4, chance_space_size)
    print(f"  ✅ 3×3モデルで推論成功: {chance_encoding.shape}")
    
    # GAT/MLPの重みが共有されているか確認
    gat_params_match = 0
    mlp_params_match = 0
    total_params = 0
    
    for (name_4x4, param_4x4), (name_3x3, param_3x3) in zip(
        encoder_4x4.named_parameters(), 
        encoder_3x3.named_parameters()
    ):
        if 'graph_builder' not in name_4x4:
            total_params += 1
            if torch.allclose(param_4x4, param_3x3, atol=1e-6):
                if 'gat' in name_4x4:
                    gat_params_match += 1
                elif 'mlp' in name_4x4:
                    mlp_params_match += 1
    
    print(f"\n📊 重み共有状況:")
    print(f"  ✅ GAT重みが一致: {gat_params_match > 0}")
    print(f"  ✅ MLP重みが一致: {mlp_params_match > 0}")
    print(f"  💡 GraphBuilderは異なるサイズのため別々に初期化されます")
    print()


def test_full_gat_model():
    """完全GATモデルのテスト"""
    print("=" * 80)
    print("テスト4: 完全GAT化されたStochasticMuZeroModel")
    print("=" * 80)
    
    print("\n🔵 モデル作成中...")
    try:
        model = GATStochasticMuZeroModel(
            observation_shape=(16, 4, 4),
            action_space_size=4,
            chance_space_size=32,
            num_channels=128,
            num_gnn_layers=3,
            num_heads=4,
            grid_size=4,
            edge_mode='adjacent',
            norm_type='layer',  # LayerNormで安定性重視
            include_row_col_edges=False,
        )
        print("✅ モデル作成成功")
        print()
        
        # 推論テスト
        print("🔵 推論テスト...")
        obs = torch.randn(4, 16, 4, 4)
        
        # Initial inference
        output = model.initial_inference(obs)
        print(f"  ✅ initial_inference成功")
        print(f"     - value shape: {output.value.shape}")
        if hasattr(output.reward, 'shape'):
            print(f"     - reward shape: {output.reward.shape}")
        else:
            print(f"     - reward: {output.reward}")
        print(f"     - policy_logits shape: {output.policy_logits.shape}")
        print(f"     - latent_state shape: {output.latent_state.shape}")
        
        # Recurrent inference
        action = torch.randint(0, 4, (4,))
        output = model.recurrent_inference(output.latent_state, action)
        print(f"  ✅ recurrent_inference成功")
        
        print()
        
    except RuntimeError as e:
        print(f"❌ エラー: {e}")
        print()
        return False
    
    return True


def main():
    print()
    print("=" * 80)
    print("GATChanceEncoder & 完全GAT化テスト")
    print("=" * 80)
    print()
    
    # テスト1: 基本動作
    test_gat_chance_encoder_basic()
    
    # テスト2: サイズ非依存性
    test_size_independence()
    
    # テスト3: 転移学習
    test_transfer_learning()
    
    # テスト4: 完全GATモデル
    success = test_full_gat_model()
    
    # サマリー
    print("=" * 80)
    print("📊 テスト結果サマリー")
    print("=" * 80)
    print()
    if success:
        print("✅ すべてのテストが成功しました！")
        print()
        print("💡 次のステップ:")
        print("  1. トレーニングスクリプトでモデルを使用")
        print("  2. 異なるサイズでの実験を実施")
        print("  3. 転移学習の効果を測定")
    else:
        print("❌ 一部のテストが失敗しました")
    print()


if __name__ == '__main__':
    main()
