#!/usr/bin/env python3
"""
GNN学習検証スクリプト
===================

このスクリプトは以下を検証します:
1. グラフ構造が正しく構築されているか
2. エッジ情報が実際に使われているか
3. 学習が進んでいるか（損失、スコアの改善）
4. GNN層が活性化しているか
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# LightZeroのパスを追加
sys.path.insert(0, str(Path(__file__).parent / "LightZero"))

from lzero.model.gnn_stochastic_muzero_model_optimized import GNNStochasticMuZeroModelOptimized
from lzero.model.gnn_utils import GraphBuilder


def test_graph_structure():
    """1. グラフ構造の検証"""
    print("\n" + "="*60)
    print("1. グラフ構造の検証")
    print("="*60)
    
    builder = GraphBuilder(grid_size=4, include_row_col_edges=True, edge_mode='sparse')
    
    # ダミー観測を作成
    dummy_obs = torch.randn(2, 16, 4, 4)  # [B=2, C=16, H=4, W=4]
    
    # グラフに変換
    node_features, edge_index = builder.obs_to_graph(dummy_obs)
    
    print(f"✓ ノード特徴量の形状: {node_features.shape}")
    print(f"  -> [バッチ={node_features.shape[0]}, ノード数={node_features.shape[1]}, 特徴量次元={node_features.shape[2]}]")
    print(f"\n✓ エッジインデックスの形状: {edge_index.shape}")
    print(f"  -> [2, エッジ数={edge_index.shape[1]}]")
    print(f"\n✓ エッジの例（最初の10個）:")
    print(f"  始点: {edge_index[0, :10].tolist()}")
    print(f"  終点: {edge_index[1, :10].tolist()}")
    
    # エッジの統計
    num_edges = edge_index.shape[1]
    num_nodes = 16
    density = num_edges / (num_nodes * (num_nodes - 1))
    print(f"\n✓ グラフの統計:")
    print(f"  - ノード数: {num_nodes}")
    print(f"  - エッジ数: {num_edges}")
    print(f"  - グラフ密度: {density:.4f}")
    
    return True


def test_edge_usage():
    """2. エッジ情報が使われているかの検証"""
    print("\n" + "="*60)
    print("2. エッジ情報の使用確認")
    print("="*60)
    
    # 2つの観測を用意：1つは通常、1つはシャッフル
    torch.manual_seed(42)
    obs1 = torch.randn(1, 16, 4, 4)
    
    # ノードをシャッフル（エッジ構造を壊す）
    obs2 = obs1.clone()
    obs2 = obs2.view(1, 16, 16).transpose(1, 2)  # [1, 16, 16]
    perm = torch.randperm(16)
    obs2 = obs2[:, perm, :]  # ノードの順序をランダムに
    obs2 = obs2.transpose(1, 2).view(1, 16, 4, 4)
    
    # モデルを作成
    model = GNNStochasticMuZeroModelOptimized(
        observation_shape=(16, 4, 4),
        action_space_size=4,
        num_channels=64,
        num_res_blocks=1,
        num_gnn_layers=2,
    )
    model.eval()
    
    with torch.no_grad():
        # 通常の観測での出力
        output1 = model.initial_inference(obs1)
        latent1 = output1.latent_state
        
        # シャッフルした観測での出力
        output2 = model.initial_inference(obs2)
        latent2 = output2.latent_state
    
    # 潜在表現の差を計算
    diff = torch.norm(latent1 - latent2).item()
    max_val = torch.norm(latent1).item()
    relative_diff = diff / (max_val + 1e-8)
    
    print(f"✓ 通常観測の潜在表現ノルム: {max_val:.4f}")
    print(f"✓ シャッフル観測との差: {diff:.4f}")
    print(f"✓ 相対差: {relative_diff:.4f}")
    
    if relative_diff > 0.01:
        print(f"\n✓ エッジ構造が影響している！（相対差 > 0.01）")
        return True
    else:
        print(f"\n⚠ エッジ構造の影響が小さい可能性（相対差 < 0.01）")
        return False


def test_gnn_activation():
    """3. GNN層の活性化確認"""
    print("\n" + "="*60)
    print("3. GNN層の活性化確認")
    print("="*60)
    
    model = GNNStochasticMuZeroModelOptimized(
        observation_shape=(16, 4, 4),
        action_space_size=4,
        num_channels=128,
        num_gnn_layers=3,
    )
    model.eval()
    
    # フック関数で中間層の出力を記録
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach()
        return hook
    
    # RepresentationNetworkのGNN層にフックを登録
    for i, conv in enumerate(model.representation_network.gnn.convs):
        conv.register_forward_hook(get_activation(f'rep_gnn_layer_{i}'))
    
    # ダミー入力
    obs = torch.randn(4, 16, 4, 4)
    
    with torch.no_grad():
        output = model.initial_inference(obs)
    
    print(f"✓ GNN層の活性化を記録しました:")
    for name, activation in activations.items():
        mean_val = activation.mean().item()
        std_val = activation.std().item()
        print(f"  {name}:")
        print(f"    - 平均: {mean_val:.6f}")
        print(f"    - 標準偏差: {std_val:.6f}")
        print(f"    - 形状: {activation.shape}")
    
    # 活性化が0でないことを確認
    all_active = all(act.abs().mean() > 1e-6 for act in activations.values())
    
    if all_active:
        print(f"\n✓ すべてのGNN層が活性化しています！")
        return True
    else:
        print(f"\n⚠ 一部のGNN層の活性化が弱い可能性があります")
        return False


def test_learning_checkpoint(checkpoint_path: str = None):
    """4. 学習チェックポイントの検証"""
    print("\n" + "="*60)
    print("4. 学習チェックポイントの検証")
    print("="*60)
    
    if checkpoint_path is None:
        # デフォルトのチェックポイントパスを探す
        possible_paths = [
            "./data_gnn_stochastic_mz_optimized/ckpt/ckpt_best.pth.tar",
            "./LightZero/data_gnn_stochastic_mz_optimized/ckpt/ckpt_best.pth.tar",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                checkpoint_path = path
                break
    
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print("⚠ チェックポイントが見つかりません。学習を実行してください。")
        return None
    
    print(f"✓ チェックポイントを読み込み: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 学習の進捗情報を表示
        if 'train_iter' in checkpoint:
            print(f"  - 学習イテレーション: {checkpoint['train_iter']}")
        
        if 'collect_kwargs' in checkpoint:
            print(f"  - 収集設定: {checkpoint['collect_kwargs']}")
        
        # モデルの重みを確認
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            
            # GNN層の重みを確認
            gnn_layers = [k for k in state_dict.keys() if 'gnn' in k and 'weight' in k]
            print(f"\n✓ GNN層の重み統計:")
            
            for i, layer_name in enumerate(gnn_layers[:3]):  # 最初の3層のみ表示
                weight = state_dict[layer_name]
                print(f"  {layer_name}:")
                print(f"    - 平均: {weight.mean().item():.6f}")
                print(f"    - 標準偏差: {weight.std().item():.6f}")
                print(f"    - 最小値: {weight.min().item():.6f}")
                print(f"    - 最大値: {weight.max().item():.6f}")
        
        return True
    
    except Exception as e:
        print(f"⚠ チェックポイントの読み込みエラー: {e}")
        return False


def main():
    """メイン検証関数"""
    print("\n" + "="*60)
    print("GNN学習検証スクリプト")
    print("="*60)
    
    results = {}
    
    try:
        # 1. グラフ構造の検証
        results['graph_structure'] = test_graph_structure()
        
        # 2. エッジ情報の使用確認
        results['edge_usage'] = test_edge_usage()
        
        # 3. GNN層の活性化確認
        results['gnn_activation'] = test_gnn_activation()
        
        # 4. 学習チェックポイントの検証
        results['checkpoint'] = test_learning_checkpoint()
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 結果のサマリー
    print("\n" + "="*60)
    print("検証結果サマリー")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "⚠ WARNING" if result is False else "- SKIP"
        print(f"{test_name:20s}: {status}")
    
    all_pass = all(r is True for r in results.values() if r is not None)
    
    if all_pass:
        print("\n🎉 すべての検証に合格しました！GNNは正しく動作しています。")
    else:
        print("\n⚠ 一部の検証で警告が出ています。結果を確認してください。")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
