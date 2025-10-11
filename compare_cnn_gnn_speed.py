"""
CNNとGNNの推論速度を比較するスクリプト

GNNが遅い理由:
1. メッセージパッシングのforループ (Pythonループで各ノードを処理)
2. グラフ構造の構築オーバーヘッド
3. 複数のGNN層 (デフォルト3層)
4. より大きな隠れ層次元 (128 vs 64)

CNNが速い理由:
1. 畳み込み演算が高度に最適化されている (CuDNN)
2. 並列計算が効率的
3. シンプルなアーキテクチャ
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('./LightZero')


def benchmark_inference_speed(model, obs, num_iterations=100):
    """推論速度をベンチマーク"""
    device = next(model.parameters()).device
    
    # ウォームアップ
    with torch.no_grad():
        for _ in range(10):
            _ = model(obs)
    
    # 計測
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(obs)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    return avg_time * 1000  # ms


def load_cnn_model(model_path, device='cuda'):
    """CNNモデルをロード"""
    from cnn_any_size_emulator import RepresentationNetwork, PolicyHead
    
    rep_net = RepresentationNetwork(
        observation_shape=(16, 4, 4),
        num_res_blocks=1,
        num_channels=64,
    ).to(device)
    
    policy_head = PolicyHead(
        input_shape=(64, 4, 4),
        action_space_size=4,
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    rep_state_dict = {}
    policy_state_dict = {}
    
    for key, value in state_dict.items():
        if 'representation_network' in key:
            new_key = key.replace('representation_network.', '')
            rep_state_dict[new_key] = value
        elif 'prediction_network.policy_head' in key:
            new_key = key.replace('prediction_network.policy_head.', '')
            policy_state_dict[new_key] = value
    
    rep_net.load_state_dict(rep_state_dict, strict=False)
    policy_head.load_state_dict(policy_state_dict, strict=False)
    
    rep_net.eval()
    policy_head.eval()
    
    return rep_net, policy_head


def load_gnn_model(model_path, device='cuda'):
    """GNNモデルをロード"""
    from gnn_any_size_emulator import GNNRepresentationNetwork, GNNPolicyHead
    
    rep_net = GNNRepresentationNetwork(
        observation_shape=(16, 4, 4),
        num_channels=128,
        num_gnn_layers=3,
        grid_size=4,
    ).to(device)
    
    policy_head = GNNPolicyHead(
        num_channels=128,
        action_space_size=4,
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    rep_state_dict = {}
    policy_state_dict = {}
    
    for key, value in state_dict.items():
        if 'representation_network' in key:
            new_key = key.replace('representation_network.', '')
            rep_state_dict[new_key] = value
        elif 'prediction_network.policy_head' in key:
            new_key = key.replace('prediction_network.policy_head.', '')
            policy_state_dict[new_key] = value
    
    rep_net.load_state_dict(rep_state_dict, strict=False)
    policy_head.load_state_dict(policy_state_dict, strict=False)
    
    rep_net.eval()
    policy_head.eval()
    
    return rep_net, policy_head


def count_parameters(model):
    """パラメータ数をカウント"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("="*70)
    print("CNN vs GNN 推論速度比較")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"デバイス: {device}\n")
    
    # テスト用の観測データ
    batch_size = 1
    obs = torch.randn(batch_size, 16, 4, 4).to(device)
    
    # CNNモデル
    print("CNNモデルをロード中...")
    cnn_model_path = "/opendilab/2048GNN/game_2048_npct-2_stochastic_muzero_ns100_upc200_rer0.0_bs512_chance-True_sslw2_seed0_250729_140944/ckpt/iteration_80000.pth.tar"
    cnn_rep, cnn_policy = load_cnn_model(cnn_model_path, device)
    
    cnn_rep_params = count_parameters(cnn_rep)
    cnn_policy_params = count_parameters(cnn_policy)
    print(f"  Representation Network: {cnn_rep_params:,} パラメータ")
    print(f"  Policy Head: {cnn_policy_params:,} パラメータ")
    print(f"  合計: {cnn_rep_params + cnn_policy_params:,} パラメータ\n")
    
    # GNNモデル
    print("GNNモデルをロード中...")
    gnn_model_path = "/opendilab/2048GNN/LightZero/data_gnn_stochastic_mz_optimized/game_2048_gnn_opt_npct-2_ns100_upc200_rer0.0_bs512_gnn3L128D_sparse_seed0_251011_032638/ckpt/ckpt_best.pth.tar"
    try:
        gnn_rep, gnn_policy = load_gnn_model(gnn_model_path, device)
        
        gnn_rep_params = count_parameters(gnn_rep)
        gnn_policy_params = count_parameters(gnn_policy)
        print(f"  Representation Network: {gnn_rep_params:,} パラメータ")
        print(f"  Policy Head: {gnn_policy_params:,} パラメータ")
        print(f"  合計: {gnn_rep_params + gnn_policy_params:,} パラメータ\n")
    except Exception as e:
        print(f"  警告: GNNモデルのロードに失敗: {e}\n")
        gnn_rep = None
        gnn_policy = None
    
    # 速度ベンチマーク
    print("="*70)
    print("推論速度ベンチマーク (100回の平均)")
    print("="*70)
    
    # CNN Representation
    print("\n[CNN Representation Network]")
    cnn_rep_time = benchmark_inference_speed(cnn_rep, obs)
    print(f"  平均時間: {cnn_rep_time:.3f} ms")
    
    # CNN full pipeline
    print("\n[CNN Full Pipeline (Representation + Policy)]")
    class CNNFullPipeline(nn.Module):
        def __init__(self, rep, policy):
            super().__init__()
            self.rep = rep
            self.policy = policy
        
        def forward(self, x):
            latent = self.rep(x)
            policy_out = self.policy(latent)
            return policy_out
    
    cnn_full = CNNFullPipeline(cnn_rep, cnn_policy)
    cnn_full_time = benchmark_inference_speed(cnn_full, obs)
    print(f"  平均時間: {cnn_full_time:.3f} ms")
    
    if gnn_rep is not None and gnn_policy is not None:
        # GNN Representation
        print("\n[GNN Representation Network]")
        gnn_rep_time = benchmark_inference_speed(gnn_rep, obs)
        print(f"  平均時間: {gnn_rep_time:.3f} ms")
        
        # GNN full pipeline
        print("\n[GNN Full Pipeline (Representation + Policy)]")
        class GNNFullPipeline(nn.Module):
            def __init__(self, rep, policy):
                super().__init__()
                self.rep = rep
                self.policy = policy
            
            def forward(self, x):
                latent = self.rep(x)
                policy_out = self.policy(latent)
                return policy_out
        
        gnn_full = GNNFullPipeline(gnn_rep, gnn_policy)
        gnn_full_time = benchmark_inference_speed(gnn_full, obs)
        print(f"  平均時間: {gnn_full_time:.3f} ms")
        
        # 比較
        print("\n" + "="*70)
        print("速度比較")
        print("="*70)
        print(f"CNN Full Pipeline:  {cnn_full_time:.3f} ms")
        print(f"GNN Full Pipeline:  {gnn_full_time:.3f} ms")
        print(f"速度差:             {gnn_full_time / cnn_full_time:.2f}x (GNN が遅い)")
        print(f"\nGNN が遅い理由:")
        print(f"  1. メッセージパッシングの Python forループ (非効率)")
        print(f"  2. グラフ構造構築のオーバーヘッド")
        print(f"  3. より深い層 (CNN: 1層, GNN: 3層)")
        print(f"  4. より大きな隠れ層 (CNN: 64次元, GNN: 128次元)")
        print(f"  5. パラメータ数 (CNN: {cnn_rep_params + cnn_policy_params:,}, GNN: {gnn_rep_params + gnn_policy_params:,})")
        
        # 1エピソードあたりの推定時間
        avg_moves_per_episode = 200  # 平均手数
        print(f"\n1エピソードあたりの推定時間 (平均{avg_moves_per_episode}手):")
        print(f"  CNN: {cnn_full_time * avg_moves_per_episode / 1000:.2f} 秒")
        print(f"  GNN: {gnn_full_time * avg_moves_per_episode / 1000:.2f} 秒")
    
    print("\n" + "="*70)
    print("最適化の提案:")
    print("="*70)
    print("GNNの速度を改善するには:")
    print("  1. メッセージパッシングをベクトル化 (forループを除去)")
    print("  2. PyTorch Geometric などの最適化されたライブラリを使用")
    print("  3. GNN層数を削減 (3層 → 2層)")
    print("  4. 隠れ層次元を削減 (128 → 96 or 64)")
    print("  5. スパースグラフ構造を活用")
    print("="*70)


if __name__ == '__main__':
    main()
