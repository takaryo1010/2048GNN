"""
GNNモデルがトレーニングで正しく使用されていることを検証するスクリプト
- CNNではなくGNN（GraphSAGE）を使用していることを証明
- 各ネットワークコンポーネントの構造を確認
- 実際のforward passでの動作を検証
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'LightZero'))

import torch
import torch.nn as nn
from easydict import EasyDict
from typing import Dict, List, Tuple
import json


def check_module_has_conv2d(module: nn.Module, module_name: str = "model") -> Tuple[bool, List[str]]:
    """
    モジュール内にConv2dレイヤーが存在するかチェック
    
    Returns:
        (has_conv2d, conv2d_layers): Conv2dが存在するか、Conv2dレイヤーのリスト
    """
    conv2d_layers = []
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Conv2d):
            conv2d_layers.append(f"{module_name}.{name}")
    
    return len(conv2d_layers) > 0, conv2d_layers


def check_module_has_gnn(module: nn.Module, module_name: str = "model") -> Tuple[bool, List[str]]:
    """
    モジュール内にGNN関連レイヤーが存在するかチェック
    
    Returns:
        (has_gnn, gnn_layers): GNNレイヤーが存在するか、GNNレイヤーのリスト
    """
    gnn_layers = []
    gnn_keywords = ['GraphSAGE', 'GNN', 'GraphBuilder', 'GAT', 'GATConv']
    
    for name, layer in module.named_modules():
        layer_type = type(layer).__name__
        if any(keyword in layer_type for keyword in gnn_keywords):
            gnn_layers.append(f"{module_name}.{name} ({layer_type})")
    
    return len(gnn_layers) > 0, gnn_layers


def analyze_model_architecture(model: nn.Module) -> Dict:
    """
    モデルのアーキテクチャを詳細に解析
    """
    analysis = {
        "model_type": type(model).__name__,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "has_representation_network": hasattr(model, 'representation_network'),
        "has_dynamics_network": hasattr(model, 'dynamics_network'),
        "has_prediction_network": hasattr(model, 'prediction_network'),
        "has_afterstate_dynamics_network": hasattr(model, 'afterstate_dynamics_network'),
        "has_afterstate_prediction_network": hasattr(model, 'afterstate_prediction_network'),
    }
    
    # 各ネットワークの解析
    networks = {}
    
    # Representation Network
    if hasattr(model, 'representation_network'):
        repr_net = model.representation_network
        has_conv, conv_layers = check_module_has_conv2d(repr_net, "representation_network")
        has_gnn, gnn_layers = check_module_has_gnn(repr_net, "representation_network")
        
        networks['representation_network'] = {
            "type": type(repr_net).__name__,
            "has_conv2d": has_conv,
            "conv2d_layers": conv_layers,
            "has_gnn": has_gnn,
            "gnn_layers": gnn_layers,
            "parameters": sum(p.numel() for p in repr_net.parameters()),
        }
    
    # Dynamics Network
    if hasattr(model, 'dynamics_network'):
        dyn_net = model.dynamics_network
        has_conv, conv_layers = check_module_has_conv2d(dyn_net, "dynamics_network")
        has_gnn, gnn_layers = check_module_has_gnn(dyn_net, "dynamics_network")
        
        networks['dynamics_network'] = {
            "type": type(dyn_net).__name__,
            "has_conv2d": has_conv,
            "conv2d_layers": conv_layers,
            "has_gnn": has_gnn,
            "gnn_layers": gnn_layers,
            "parameters": sum(p.numel() for p in dyn_net.parameters()),
        }
    
    # Prediction Network (Value/Policy heads)
    if hasattr(model, 'prediction_network'):
        pred_net = model.prediction_network
        has_conv, conv_layers = check_module_has_conv2d(pred_net, "prediction_network")
        has_gnn, gnn_layers = check_module_has_gnn(pred_net, "prediction_network")
        
        networks['prediction_network'] = {
            "type": type(pred_net).__name__,
            "has_conv2d": has_conv,
            "conv2d_layers": conv_layers,
            "has_gnn": has_gnn,
            "gnn_layers": gnn_layers,
            "parameters": sum(p.numel() for p in pred_net.parameters()),
        }
    
    # Afterstate Dynamics Network
    if hasattr(model, 'afterstate_dynamics_network'):
        afterstate_dyn_net = model.afterstate_dynamics_network
        has_conv, conv_layers = check_module_has_conv2d(afterstate_dyn_net, "afterstate_dynamics_network")
        has_gnn, gnn_layers = check_module_has_gnn(afterstate_dyn_net, "afterstate_dynamics_network")
        
        networks['afterstate_dynamics_network'] = {
            "type": type(afterstate_dyn_net).__name__,
            "has_conv2d": has_conv,
            "conv2d_layers": conv_layers,
            "has_gnn": has_gnn,
            "gnn_layers": gnn_layers,
            "parameters": sum(p.numel() for p in afterstate_dyn_net.parameters()),
        }
    
    # Afterstate Prediction Network
    if hasattr(model, 'afterstate_prediction_network'):
        afterstate_pred_net = model.afterstate_prediction_network
        has_conv, conv_layers = check_module_has_conv2d(afterstate_pred_net, "afterstate_prediction_network")
        has_gnn, gnn_layers = check_module_has_gnn(afterstate_pred_net, "afterstate_prediction_network")
        
        networks['afterstate_prediction_network'] = {
            "type": type(afterstate_pred_net).__name__,
            "has_conv2d": has_conv,
            "conv2d_layers": conv_layers,
            "has_gnn": has_gnn,
            "gnn_layers": gnn_layers,
            "parameters": sum(p.numel() for p in afterstate_pred_net.parameters()),
        }
    
    analysis['networks'] = networks
    
    return analysis


def test_forward_pass_with_hooks(model: nn.Module, batch_size: int = 4) -> Dict:
    """
    Forward passを実行し、フックを使って各レイヤーの動作を追跡
    """
    model.eval()
    
    # フックで追跡する情報
    activations = {}
    layer_types = {}
    
    def get_activation(name):
        def hook(module, input, output):
            layer_type = type(module).__name__
            layer_types[name] = layer_type
            
            # 出力の形状を記録
            if isinstance(output, torch.Tensor):
                activations[name] = {
                    "shape": list(output.shape),
                    "dtype": str(output.dtype),
                    "device": str(output.device),
                    "layer_type": layer_type
                }
            elif isinstance(output, tuple):
                activations[name] = {
                    "shapes": [list(o.shape) if isinstance(o, torch.Tensor) else None for o in output],
                    "layer_type": layer_type
                }
        return hook
    
    # すべてのレイヤーにフックを登録
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            hooks.append(module.register_forward_hook(get_activation(name)))
    
    # ダミー入力でforward pass
    try:
        with torch.no_grad():
            dummy_obs = torch.randn(batch_size, 16, 4, 4)
            
            # Representation network
            if hasattr(model, 'representation_network'):
                print("\n🔍 Testing Representation Network...")
                latent_state = model.representation_network(dummy_obs)
                print(f"   Input: {dummy_obs.shape} -> Output: {latent_state.shape}")
            
            # Dynamics network (需要action)
            if hasattr(model, 'dynamics_network'):
                print("\n🔍 Testing Dynamics Network...")
                dummy_action = torch.randint(0, 4, (batch_size,))
                # One-hot encode action
                dummy_action_encoded = torch.zeros(batch_size, 4)
                dummy_action_encoded.scatter_(1, dummy_action.unsqueeze(1), 1)
                
                if hasattr(model, 'representation_network'):
                    next_latent, reward = model.dynamics_network(latent_state, dummy_action_encoded)
                    print(f"   State: {latent_state.shape}, Action: {dummy_action_encoded.shape}")
                    print(f"   -> Next State: {next_latent.shape}, Reward: {reward.shape}")
    
    finally:
        # フックを削除
        for hook in hooks:
            hook.remove()
    
    # GNN関連レイヤーの動作を確認
    gnn_activations = {}
    conv2d_activations = {}
    
    for name, info in activations.items():
        layer_type = info.get('layer_type', '')
        if any(kw in layer_type for kw in ['GraphSAGE', 'GNN', 'GraphBuilder', 'GAT']):
            gnn_activations[name] = info
        elif 'Conv2d' in layer_type:
            conv2d_activations[name] = info
    
    return {
        "total_layers_activated": len(activations),
        "gnn_layers_activated": len(gnn_activations),
        "conv2d_layers_activated": len(conv2d_activations),
        "gnn_activations": gnn_activations,
        "conv2d_activations": conv2d_activations,
    }


def print_verification_report(analysis: Dict, forward_test: Dict):
    """
    検証レポートを出力
    """
    print("\n" + "="*80)
    print("GNN MODEL VERIFICATION REPORT")
    print("GNNモデル検証レポート")
    print("="*80)
    
    print(f"\n📊 モデル基本情報:")
    print(f"   - モデルタイプ: {analysis['model_type']}")
    print(f"   - 総パラメータ数: {analysis['total_parameters']:,}")
    print(f"   - 訓練可能パラメータ数: {analysis['trainable_parameters']:,}")
    
    print(f"\n🏗️ ネットワーク構成:")
    print(f"   - Representation Network: {'✓' if analysis['has_representation_network'] else '✗'}")
    print(f"   - Dynamics Network: {'✓' if analysis['has_dynamics_network'] else '✗'}")
    print(f"   - Prediction Network: {'✓' if analysis['has_prediction_network'] else '✗'}")
    print(f"   - Afterstate Dynamics Network: {'✓' if analysis['has_afterstate_dynamics_network'] else '✗'}")
    print(f"   - Afterstate Prediction Network: {'✓' if analysis['has_afterstate_prediction_network'] else '✗'}")
    
    # 各ネットワークの詳細
    print("\n" + "-"*80)
    print("📝 各ネットワークの詳細解析:")
    print("-"*80)
    
    total_conv2d = 0
    total_gnn = 0
    
    for net_name, net_info in analysis['networks'].items():
        print(f"\n🔹 {net_name}:")
        print(f"   タイプ: {net_info['type']}")
        print(f"   パラメータ数: {net_info['parameters']:,}")
        
        # CNN check
        if net_info['has_conv2d']:
            print(f"   ⚠️  Conv2dレイヤー検出: {len(net_info['conv2d_layers'])}個")
            for layer in net_info['conv2d_layers']:
                print(f"      - {layer}")
            total_conv2d += len(net_info['conv2d_layers'])
        else:
            print(f"   ✓  Conv2dレイヤー: なし (CNN不使用)")
        
        # GNN check
        if net_info['has_gnn']:
            print(f"   ✓  GNNレイヤー検出: {len(net_info['gnn_layers'])}個")
            for layer in net_info['gnn_layers']:
                print(f"      - {layer}")
            total_gnn += len(net_info['gnn_layers'])
        else:
            print(f"   ⚠️  GNNレイヤー: なし")
    
    # Forward pass結果
    print("\n" + "-"*80)
    print("🚀 Forward Pass実行結果:")
    print("-"*80)
    print(f"   - 動作したレイヤー総数: {forward_test['total_layers_activated']}")
    print(f"   - GNNレイヤー動作数: {forward_test['gnn_layers_activated']}")
    print(f"   - Conv2dレイヤー動作数: {forward_test['conv2d_layers_activated']}")
    
    if forward_test['gnn_activations']:
        print(f"\n   ✓ GNNレイヤーの動作を確認:")
        for name, info in list(forward_test['gnn_activations'].items())[:5]:  # 最初の5つ
            print(f"      - {name} ({info['layer_type']})")
    
    if forward_test['conv2d_activations']:
        print(f"\n   ⚠️  Conv2dレイヤーの動作を検出:")
        for name, info in forward_test['conv2d_activations'].items():
            print(f"      - {name} ({info['layer_type']})")
    
    # 最終判定
    print("\n" + "="*80)
    print("🎯 最終検証結果:")
    print("="*80)
    
    checks_passed = []
    checks_failed = []
    
    # Check 1: GNNレイヤーの存在
    if total_gnn > 0:
        checks_passed.append("✓ GNNレイヤーが検出されました")
    else:
        checks_failed.append("✗ GNNレイヤーが検出されませんでした")
    
    # Check 2: CNNレイヤーの不在（Chance Encoder除く）
    if total_conv2d == 0:
        checks_passed.append("✓ CNN (Conv2d) レイヤーは使用されていません")
    else:
        # Chance encoderのみの場合は許容
        if all('chance' in layer.lower() for net_name, net_info in analysis['networks'].items() 
               for layer in net_info.get('conv2d_layers', [])):
            checks_passed.append("✓ Conv2dはChance Encoderのみで使用（許容）")
        else:
            checks_failed.append(f"⚠️  Conv2dレイヤーが{total_conv2d}個検出されました")
    
    # Check 3: Forward passでGNNが動作
    if forward_test['gnn_layers_activated'] > 0:
        checks_passed.append(f"✓ Forward passでGNNレイヤーが動作しました ({forward_test['gnn_layers_activated']}個)")
    else:
        checks_failed.append("✗ Forward passでGNNレイヤーが動作しませんでした")
    
    # Check 4: モデルタイプがGNN
    if 'GNN' in analysis['model_type']:
        checks_passed.append(f"✓ モデルタイプがGNNです ({analysis['model_type']})")
    else:
        checks_failed.append(f"⚠️  モデルタイプがGNNではありません ({analysis['model_type']})")
    
    print("\n合格した検証項目:")
    for check in checks_passed:
        print(f"  {check}")
    
    if checks_failed:
        print("\n⚠️  失敗した検証項目:")
        for check in checks_failed:
            print(f"  {check}")
    
    # 総合判定
    print("\n" + "="*80)
    if len(checks_failed) == 0:
        print("🎉 すべての検証に合格しました！")
        print("✅ このモデルは正しくGNNを使用しています（CNNではありません）")
    else:
        print(f"⚠️  {len(checks_failed)}個の検証項目が失敗しました")
        print("一部の問題を確認してください")
    print("="*80 + "\n")


def main():
    """
    メイン検証ロジック
    """
    print("\n🔍 GNNモデルの検証を開始します...\n")
    
    # Load config
    from zoo.game_2048.config.stochastic_muzero_2048_gnn_config import main_config, create_config
    
    print("📋 設定ファイルを読み込みました")
    print(f"   - Model type: {create_config.model.type}")
    print(f"   - Import names: {create_config.model.import_names}")
    
    # Create model
    print("\n🏗️  モデルを構築中...")
    
    # Import model
    from lzero.model import GNNStochasticMuZeroModel
    
    # Create model instance
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
    
    print("✓ モデルの構築が完了しました")
    
    # Analyze architecture
    print("\n🔬 アーキテクチャを解析中...")
    analysis = analyze_model_architecture(model)
    
    # Test forward pass
    print("\n🚀 Forward passをテスト中...")
    forward_test = test_forward_pass_with_hooks(model, batch_size=4)
    
    # Print report
    print_verification_report(analysis, forward_test)
    
    # Save results
    results = {
        "architecture_analysis": analysis,
        "forward_pass_test": forward_test,
        "config": {
            "model_type": create_config.model.type,
            "num_gnn_layers": model_config.num_gnn_layers,
            "num_channels": model_config.num_channels,
            "edge_mode": model_config.edge_mode,
            "grid_size": model_config.grid_size,
        }
    }
    
    output_file = "gnn_verification_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"📁 検証結果を {output_file} に保存しました\n")


if __name__ == "__main__":
    main()
