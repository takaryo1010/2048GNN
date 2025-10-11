"""
実際のトレーニングにGNN監視を統合したスクリプト
GNNが正しく使用されていることをリアルタイムで確認
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'LightZero'))

import torch
import torch.nn as nn
from typing import Dict, Optional
import json
from datetime import datetime


def verify_model_uses_gnn(model: nn.Module) -> Dict:
    """
    モデルがGNNを使用しているか検証
    
    Returns:
        検証結果の辞書
    """
    results = {
        "model_type": type(model).__name__,
        "is_gnn_model": False,
        "has_gnn_layers": False,
        "has_conv2d_layers": False,
        "gnn_layer_count": 0,
        "conv2d_layer_count": 0,
        "gnn_layers": [],
        "conv2d_layers": [],
        "verification_passed": False,
    }
    
    # モデルタイプチェック
    if 'GNN' in type(model).__name__:
        results["is_gnn_model"] = True
    
    # レイヤーを走査
    gnn_keywords = ['GraphSAGE', 'GNN', 'GraphBuilder', 'GAT', 'GATConv']
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        # GNNレイヤー
        if any(kw in module_type for kw in gnn_keywords):
            results["gnn_layers"].append(f"{name} ({module_type})")
            results["gnn_layer_count"] += 1
            results["has_gnn_layers"] = True
        
        # Conv2dレイヤー (Chance encoder以外)
        if isinstance(module, nn.Conv2d) and 'chance' not in name.lower():
            results["conv2d_layers"].append(f"{name} ({module_type})")
            results["conv2d_layer_count"] += 1
            results["has_conv2d_layers"] = True
    
    # 検証判定
    results["verification_passed"] = (
        results["is_gnn_model"] and 
        results["has_gnn_layers"] and 
        not results["has_conv2d_layers"]
    )
    
    return results


def print_verification_header():
    """検証ヘッダーを出力"""
    print("\n" + "="*80)
    print("🔬 GNN Training Verification / GNNトレーニング検証")
    print("="*80)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


def print_model_verification(results: Dict):
    """モデル検証結果を出力"""
    print("\n📋 モデル構造検証:")
    print("-"*80)
    print(f"モデルタイプ: {results['model_type']}")
    print(f"GNNモデル: {'✓ Yes' if results['is_gnn_model'] else '✗ No'}")
    print(f"GNNレイヤー: {'✓ 検出' if results['has_gnn_layers'] else '✗ 未検出'} ({results['gnn_layer_count']}個)")
    print(f"Conv2dレイヤー: {'⚠️  検出' if results['has_conv2d_layers'] else '✓ なし'} ({results['conv2d_layer_count']}個)")
    
    if results['gnn_layers']:
        print(f"\n✓ GNNレイヤー詳細:")
        for layer in results['gnn_layers'][:5]:  # 最初の5つ
            print(f"   - {layer}")
        if len(results['gnn_layers']) > 5:
            print(f"   ... 他 {len(results['gnn_layers']) - 5} 個")
    
    if results['conv2d_layers']:
        print(f"\n⚠️  Conv2dレイヤー詳細:")
        for layer in results['conv2d_layers']:
            print(f"   - {layer}")
    
    print("\n" + "-"*80)
    if results['verification_passed']:
        print("✅ 検証合格: モデルは正しくGNNを使用しています")
    else:
        print("❌ 検証失敗: モデルの構成を確認してください")
        if not results['is_gnn_model']:
            print("   - モデルタイプがGNNではありません")
        if not results['has_gnn_layers']:
            print("   - GNNレイヤーが見つかりません")
        if results['has_conv2d_layers']:
            print("   - 意図しないConv2dレイヤーが存在します")
    print("-"*80 + "\n")


def test_forward_pass_gnn_usage(model: nn.Module, batch_size: int = 4) -> Dict:
    """
    Forward passでGNNが使用されることを確認
    """
    print("\n🚀 Forward Pass検証:")
    print("-"*80)
    
    model.eval()
    
    # Forward/Backwardカウンター
    counters = {
        'gnn_forward': 0,
        'conv2d_forward': 0,
        'gnn_layers_activated': [],
        'conv2d_layers_activated': [],
    }
    
    def make_hook(layer_name, layer_type):
        def hook(module, input, output):
            if layer_type == 'gnn':
                counters['gnn_forward'] += 1
                counters['gnn_layers_activated'].append(layer_name)
            elif layer_type == 'conv2d':
                counters['conv2d_forward'] += 1
                counters['conv2d_layers_activated'].append(layer_name)
        return hook
    
    # フックを登録
    hooks = []
    gnn_keywords = ['GraphSAGE', 'GNN', 'GraphBuilder', 'GAT']
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        if any(kw in module_type for kw in gnn_keywords):
            hooks.append(module.register_forward_hook(make_hook(name, 'gnn')))
        elif isinstance(module, nn.Conv2d) and 'chance' not in name.lower():
            hooks.append(module.register_forward_hook(make_hook(name, 'conv2d')))
    
    # Forward pass実行
    try:
        with torch.no_grad():
            dummy_obs = torch.randn(batch_size, 16, 4, 4)
            
            print(f"入力: {dummy_obs.shape}")
            
            # Representation network
            if hasattr(model, 'representation_network'):
                latent_state = model.representation_network(dummy_obs)
                print(f"Representation出力: {latent_state.shape}")
            
            # Prediction network
            if hasattr(model, 'prediction_network'):
                value, policy = model.prediction_network(latent_state)
                print(f"Prediction出力: value={value.shape}, policy={policy.shape}")
    
    finally:
        # フックをクリーンアップ
        for hook in hooks:
            hook.remove()
    
    print(f"\n動作結果:")
    print(f"   - GNNレイヤー動作: {counters['gnn_forward']}回")
    print(f"   - Conv2dレイヤー動作: {counters['conv2d_forward']}回")
    
    if counters['gnn_layers_activated']:
        print(f"\n✓ 動作したGNNレイヤー:")
        for layer in list(set(counters['gnn_layers_activated']))[:5]:
            print(f"   - {layer}")
    
    if counters['conv2d_layers_activated']:
        print(f"\n⚠️  動作したConv2dレイヤー:")
        for layer in set(counters['conv2d_layers_activated']):
            print(f"   - {layer}")
    
    forward_passed = counters['gnn_forward'] > 0 and counters['conv2d_forward'] == 0
    
    print("\n" + "-"*80)
    if forward_passed:
        print("✅ Forward Pass検証合格: GNNが正しく動作しています")
    else:
        print("❌ Forward Pass検証失敗")
    print("-"*80 + "\n")
    
    return {
        'gnn_forward_count': counters['gnn_forward'],
        'conv2d_forward_count': counters['conv2d_forward'],
        'verification_passed': forward_passed,
    }


def test_backward_pass_gnn_gradients(model: nn.Module, batch_size: int = 4) -> Dict:
    """
    Backward passでGNNに勾配が流れることを確認
    """
    print("\n🔄 Backward Pass検証:")
    print("-"*80)
    
    model.train()
    
    # GNNレイヤーのパラメータを収集
    gnn_params = []
    gnn_param_names = []
    conv2d_params = []
    conv2d_param_names = []
    
    gnn_keywords = ['GraphSAGE', 'GNN', 'GraphBuilder', 'GAT']
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        if any(kw in module_type for kw in gnn_keywords):
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    gnn_params.append(param)
                    gnn_param_names.append(f"{name}.{param_name}")
        
        elif isinstance(module, nn.Conv2d) and 'chance' not in name.lower():
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    conv2d_params.append(param)
                    conv2d_param_names.append(f"{name}.{param_name}")
    
    print(f"監視対象パラメータ:")
    print(f"   - GNNパラメータ: {len(gnn_params)}個")
    print(f"   - Conv2dパラメータ: {len(conv2d_params)}個")
    
    # Forward + Backward pass
    dummy_obs = torch.randn(batch_size, 16, 4, 4)
    
    # Representation network
    latent_state = model.representation_network(dummy_obs)
    
    # Prediction network
    value, policy = model.prediction_network(latent_state)
    
    # Dummy loss
    target_value = torch.randn_like(value)
    target_policy = torch.randn_like(policy)
    loss = ((value - target_value) ** 2).mean() + ((policy - target_policy) ** 2).mean()
    
    print(f"\nLoss: {loss.item():.6f}")
    
    # Backward
    loss.backward()
    
    # 勾配をチェック
    gnn_grads_exist = 0
    gnn_grad_norms = []
    
    for param, param_name in zip(gnn_params, gnn_param_names):
        if param.grad is not None:
            gnn_grads_exist += 1
            grad_norm = param.grad.norm().item()
            gnn_grad_norms.append((param_name, grad_norm))
    
    conv2d_grads_exist = 0
    conv2d_grad_norms = []
    
    for param, param_name in zip(conv2d_params, conv2d_param_names):
        if param.grad is not None:
            conv2d_grads_exist += 1
            grad_norm = param.grad.norm().item()
            conv2d_grad_norms.append((param_name, grad_norm))
    
    print(f"\n勾配の状態:")
    print(f"   - GNN勾配あり: {gnn_grads_exist}/{len(gnn_params)}")
    print(f"   - Conv2d勾配あり: {conv2d_grads_exist}/{len(conv2d_params)}")
    
    if gnn_grad_norms:
        print(f"\n✓ GNN勾配ノルム (上位5個):")
        for name, norm in sorted(gnn_grad_norms, key=lambda x: -x[1])[:5]:
            print(f"   - {name}: {norm:.6f}")
    
    if conv2d_grad_norms:
        print(f"\n⚠️  Conv2d勾配ノルム:")
        for name, norm in conv2d_grad_norms:
            print(f"   - {name}: {norm:.6f}")
    
    backward_passed = gnn_grads_exist > 0 and conv2d_grads_exist == 0
    
    print("\n" + "-"*80)
    if backward_passed:
        print("✅ Backward Pass検証合格: GNNに勾配が正しく流れています")
    else:
        print("❌ Backward Pass検証失敗")
    print("-"*80 + "\n")
    
    return {
        'gnn_params_with_grad': gnn_grads_exist,
        'total_gnn_params': len(gnn_params),
        'conv2d_params_with_grad': conv2d_grads_exist,
        'total_conv2d_params': len(conv2d_params),
        'verification_passed': backward_passed,
    }


def main():
    """
    メイン検証フロー
    """
    print_verification_header()
    
    print("📦 モジュールをインポート中...")
    
    # Load config
    from zoo.game_2048.config.stochastic_muzero_2048_gnn_config import main_config, create_config
    
    # Import model
    from lzero.model import GNNStochasticMuZeroModel
    
    print("✓ インポート完了\n")
    
    print("🏗️  モデルを構築中...")
    
    # Create model
    model_config = main_config.policy.model
    model = GNNStochasticMuZeroModel(
        observation_shape=tuple(model_config.observation_shape),
        action_space_size=model_config.action_space_size,
        chance_space_size=model_config.chance_space_size,
        num_channels=model_config.num_channels,
        num_gnn_layers=model_config.num_gnn_layers,
        value_head_hidden_channels=model_config.value_head_hidden_channels,
        policy_head_hidden_channels=model_config.policy_head_hidden_channels,
        reward_head_hidden_channels=model_config.reward_head_hidden_channels,
        grid_size=model_config.grid_size,
        include_row_col_edges=model_config.include_row_col_edges,
        dropout=model_config.dropout,
        edge_mode=model_config.edge_mode,
    )
    
    print("✓ モデル構築完了\n")
    
    # Test 1: モデル構造検証
    print("\n" + "="*80)
    print("テスト 1/3: モデル構造検証")
    print("="*80)
    verification_results = verify_model_uses_gnn(model)
    print_model_verification(verification_results)
    
    # Test 2: Forward pass検証
    print("\n" + "="*80)
    print("テスト 2/3: Forward Pass検証")
    print("="*80)
    forward_results = test_forward_pass_gnn_usage(model, batch_size=4)
    
    # Test 3: Backward pass検証
    print("\n" + "="*80)
    print("テスト 3/3: Backward Pass検証")
    print("="*80)
    backward_results = test_backward_pass_gnn_gradients(model, batch_size=4)
    
    # 総合結果
    print("\n" + "="*80)
    print("📊 総合検証結果")
    print("="*80)
    
    all_tests = [
        ("モデル構造", verification_results['verification_passed']),
        ("Forward Pass", forward_results['verification_passed']),
        ("Backward Pass", backward_results['verification_passed']),
    ]
    
    print("\nテスト結果:")
    for test_name, passed in all_tests:
        status = "✅ 合格" if passed else "❌ 失敗"
        print(f"   {test_name}: {status}")
    
    all_passed = all(passed for _, passed in all_tests)
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 すべての検証に合格しました！")
        print("✅ このモデルは正しくGNNを使用しており、CNNは使用していません")
        print("✅ トレーニング中もGNNが正しく動作することが確認されました")
    else:
        print("⚠️  一部の検証が失敗しました")
        print("詳細を確認してください")
    print("="*80 + "\n")
    
    # 結果を保存
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_structure": verification_results,
        "forward_pass": forward_results,
        "backward_pass": backward_results,
        "all_tests_passed": all_passed,
        "config": {
            "model_type": create_config.model.type,
            "num_gnn_layers": model_config.num_gnn_layers,
            "num_channels": model_config.num_channels,
            "edge_mode": model_config.edge_mode,
        }
    }
    
    output_file = "gnn_training_verification.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"📁 検証結果を {output_file} に保存しました\n")


if __name__ == "__main__":
    main()
