"""
GNNモデルのBackward Pass（学習フェーズ）を詳細に検証
- GNN内部のパラメータに勾配が流れることを確認
- 実際の学習が行われていることを証明
- CNNではなくGNNが学習されていることを確認
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'LightZero'))

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import json
from datetime import datetime


def collect_gnn_parameters(model: nn.Module) -> Tuple[List, List[str]]:
    """
    GNNレイヤーのすべてのパラメータを収集
    """
    gnn_params = []
    param_names = []
    
    # より広範なキーワードで検索
    gnn_path_keywords = ['gnn', 'graph', 'sage', 'representation_network', 'dynamics_network']
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # GNN関連のパスに含まれるパラメータ
            if any(kw in name.lower() for kw in gnn_path_keywords):
                # Conv2dは除外
                if 'conv2d' not in name.lower() and 'chance' not in name.lower():
                    gnn_params.append(param)
                    param_names.append(name)
    
    return gnn_params, param_names


def collect_conv2d_parameters(model: nn.Module) -> Tuple[List, List[str]]:
    """
    Conv2dレイヤーのパラメータを収集（Chance encoder以外）
    """
    conv_params = []
    param_names = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'chance' not in name.lower():
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    full_name = f"{name}.{param_name}"
                    conv_params.append(param)
                    param_names.append(full_name)
    
    return conv_params, param_names


def test_single_backward_pass(model: nn.Module, batch_size: int = 8) -> Dict:
    """
    1回のBackward Passを実行してGNNの勾配を検証
    """
    print("\n" + "="*80)
    print("テスト1: 単一Backward Pass検証")
    print("="*80)
    
    model.train()
    
    # パラメータを収集
    gnn_params, gnn_param_names = collect_gnn_parameters(model)
    conv_params, conv_param_names = collect_conv2d_parameters(model)
    
    print(f"\n📊 パラメータ統計:")
    print(f"   - GNNパラメータ: {len(gnn_params)}個")
    print(f"   - Conv2dパラメータ: {len(conv_params)}個")
    
    if len(gnn_params) == 0:
        print("   ⚠️  GNNパラメータが見つかりません！")
        return {"status": "FAILED", "reason": "No GNN parameters found"}
    
    # Forward pass
    dummy_obs = torch.randn(batch_size, 16, 4, 4)
    
    print(f"\n🚀 Forward Pass実行中...")
    latent_state = model.representation_network(dummy_obs)
    value, policy = model.prediction_network(latent_state)
    
    print(f"   - Latent state: {latent_state.shape}")
    print(f"   - Value: {value.shape}")
    print(f"   - Policy: {policy.shape}")
    
    # Dummy loss
    target_value = torch.randn_like(value)
    target_policy = torch.randn_like(policy)
    
    loss = ((value - target_value) ** 2).mean() + ((policy - target_policy) ** 2).mean()
    
    print(f"\n📉 Loss: {loss.item():.6f}")
    
    # Backward pass
    print(f"\n⬅️  Backward Pass実行中...")
    loss.backward()
    
    # 勾配をチェック
    gnn_grads = []
    gnn_grad_stats = []
    
    for param, param_name in zip(gnn_params, gnn_param_names):
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_max = param.grad.abs().max().item()
            
            gnn_grads.append({
                'name': param_name,
                'grad_norm': grad_norm,
                'grad_mean': grad_mean,
                'grad_max': grad_max,
                'param_shape': list(param.shape),
            })
            gnn_grad_stats.append((param_name, grad_norm))
    
    conv_grads = []
    for param, param_name in zip(conv_params, conv_param_names):
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            conv_grads.append((param_name, grad_norm))
    
    print(f"\n✅ 勾配の状態:")
    print(f"   - GNN勾配あり: {len(gnn_grads)}/{len(gnn_params)}")
    print(f"   - Conv2d勾配あり: {len(conv_grads)}/{len(conv_params)}")
    
    if gnn_grads:
        print(f"\n📈 GNN勾配ノルム (上位10個):")
        for name, norm in sorted(gnn_grad_stats, key=lambda x: -x[1])[:10]:
            print(f"   - {name}: {norm:.6e}")
    
    if conv_grads:
        print(f"\n⚠️  Conv2d勾配ノルム:")
        for name, norm in conv_grads:
            print(f"   - {name}: {norm:.6e}")
    
    # 判定
    passed = len(gnn_grads) > 0 and len(conv_grads) == 0
    
    print("\n" + "-"*80)
    if passed:
        print(f"✅ Backward Pass検証: 合格")
        print(f"   - {len(gnn_grads)}個のGNNパラメータに勾配が流れました")
        print(f"   - Conv2dパラメータには勾配が流れていません")
    else:
        print(f"❌ Backward Pass検証: 失敗")
        if len(gnn_grads) == 0:
            print(f"   - GNNパラメータに勾配が流れていません")
        if len(conv_grads) > 0:
            print(f"   - Conv2dパラメータに勾配が流れました")
    print("-"*80)
    
    return {
        "status": "PASSED" if passed else "FAILED",
        "gnn_params_total": len(gnn_params),
        "gnn_params_with_grad": len(gnn_grads),
        "conv_params_total": len(conv_params),
        "conv_params_with_grad": len(conv_grads),
        "gnn_gradients": gnn_grads,
        "loss": loss.item(),
    }


def test_multi_step_training(model: nn.Module, num_steps: int = 10, batch_size: int = 8) -> Dict:
    """
    複数ステップの学習を実行してGNNが更新されることを確認
    """
    print("\n" + "="*80)
    print(f"テスト2: {num_steps}ステップの学習シミュレーション")
    print("="*80)
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # GNNパラメータを収集
    gnn_params, gnn_param_names = collect_gnn_parameters(model)
    conv_params, conv_param_names = collect_conv2d_parameters(model)
    
    print(f"\n📊 監視対象:")
    print(f"   - GNNパラメータ: {len(gnn_params)}個")
    print(f"   - Conv2dパラメータ: {len(conv_params)}個")
    
    # 初期パラメータ値を記録
    initial_params = {}
    for param, name in zip(gnn_params[:5], gnn_param_names[:5]):  # 最初の5個のみ
        initial_params[name] = param.data.clone()
    
    # 学習ループ
    print(f"\n🏋️  {num_steps}ステップの学習を実行中...")
    
    losses = []
    gnn_grad_norms = []
    conv_grad_norms = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Forward pass
        dummy_obs = torch.randn(batch_size, 16, 4, 4)
        latent_state = model.representation_network(dummy_obs)
        value, policy = model.prediction_network(latent_state)
        
        # Loss
        target_value = torch.randn_like(value)
        target_policy = torch.randn_like(policy)
        loss = ((value - target_value) ** 2).mean() + ((policy - target_policy) ** 2).mean()
        
        # Backward
        loss.backward()
        
        # 勾配ノルムを記録
        step_gnn_grads = []
        for param in gnn_params:
            if param.grad is not None:
                step_gnn_grads.append(param.grad.norm().item())
        
        step_conv_grads = []
        for param in conv_params:
            if param.grad is not None:
                step_conv_grads.append(param.grad.norm().item())
        
        gnn_grad_norms.append(sum(step_gnn_grads) / len(step_gnn_grads) if step_gnn_grads else 0.0)
        conv_grad_norms.append(sum(step_conv_grads) / len(step_conv_grads) if step_conv_grads else 0.0)
        
        # Update
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 2 == 0 or step == 0:
            print(f"   Step {step+1}/{num_steps}: Loss={loss.item():.6f}, "
                  f"GNN Grad Norm={gnn_grad_norms[-1]:.6e}, "
                  f"Conv2d Grad Norm={conv_grad_norms[-1]:.6e}")
    
    # パラメータの変化を確認
    print(f"\n📊 パラメータ変化の確認 (最初の5個):")
    param_changes = {}
    for param, name in zip(gnn_params[:5], gnn_param_names[:5]):
        if name in initial_params:
            change = (param.data - initial_params[name]).norm().item()
            param_changes[name] = change
            print(f"   - {name}: 変化量={change:.6e}")
    
    # 統計
    avg_loss = sum(losses) / len(losses)
    avg_gnn_grad = sum(gnn_grad_norms) / len(gnn_grad_norms)
    avg_conv_grad = sum(conv_grad_norms) / len(conv_grad_norms) if conv_grad_norms else 0.0
    
    print(f"\n📈 統計サマリー:")
    print(f"   - 平均Loss: {avg_loss:.6f}")
    print(f"   - 平均GNN勾配ノルム: {avg_gnn_grad:.6e}")
    print(f"   - 平均Conv2d勾配ノルム: {avg_conv_grad:.6e}")
    print(f"   - GNN勾配が発生した回数: {sum(1 for g in gnn_grad_norms if g > 0)}/{num_steps}")
    print(f"   - Conv2d勾配が発生した回数: {sum(1 for g in conv_grad_norms if g > 0)}/{num_steps}")
    
    # 判定
    gnn_params_changed = sum(1 for c in param_changes.values() if c > 1e-6)
    passed = (
        avg_gnn_grad > 0 and 
        avg_conv_grad == 0 and 
        gnn_params_changed > 0
    )
    
    print("\n" + "-"*80)
    if passed:
        print(f"✅ 学習シミュレーション: 合格")
        print(f"   - GNNパラメータが正常に更新されました")
        print(f"   - Conv2dパラメータは更新されていません")
        print(f"   - {gnn_params_changed}/{len(param_changes)}個のパラメータが変化しました")
    else:
        print(f"❌ 学習シミュレーション: 失敗")
        if avg_gnn_grad == 0:
            print(f"   - GNNに勾配が流れていません")
        if avg_conv_grad > 0:
            print(f"   - Conv2dに勾配が流れました")
        if gnn_params_changed == 0:
            print(f"   - GNNパラメータが更新されていません")
    print("-"*80)
    
    return {
        "status": "PASSED" if passed else "FAILED",
        "num_steps": num_steps,
        "avg_loss": avg_loss,
        "avg_gnn_grad_norm": avg_gnn_grad,
        "avg_conv_grad_norm": avg_conv_grad,
        "gnn_params_changed": gnn_params_changed,
        "total_params_monitored": len(param_changes),
        "losses": losses,
        "gnn_grad_norms": gnn_grad_norms,
        "conv_grad_norms": conv_grad_norms,
    }


def test_dynamics_network_backward(model: nn.Module, batch_size: int = 8) -> Dict:
    """
    Dynamics Networkの学習も確認
    """
    print("\n" + "="*80)
    print("テスト3: Dynamics Network Backward Pass検証")
    print("="*80)
    
    model.train()
    
    # Dynamics network内のGNNパラメータを収集
    dynamics_gnn_params = []
    dynamics_param_names = []
    
    if hasattr(model, 'dynamics_network'):
        for name, param in model.dynamics_network.named_parameters():
            if param.requires_grad:
                # GNN関連のパラメータ
                if 'gnn' in name.lower() or 'graph' in name.lower() or 'sage' in name.lower():
                    if 'conv2d' not in name.lower() and 'chance' not in name.lower():
                        dynamics_gnn_params.append(param)
                        dynamics_param_names.append(f"dynamics_network.{name}")
    
    print(f"\n📊 Dynamics Network GNNパラメータ: {len(dynamics_gnn_params)}個")
    
    if len(dynamics_gnn_params) == 0:
        print("   ⚠️  Dynamics NetworkにGNNパラメータが見つかりません")
        return {"status": "SKIPPED", "reason": "No dynamics GNN parameters"}
    
    # Forward pass through dynamics
    dummy_obs = torch.randn(batch_size, 16, 4, 4)
    latent_state = model.representation_network(dummy_obs)
    
    # Dynamics network forward
    dummy_action = torch.randint(0, 4, (batch_size,))
    dummy_action_encoded = torch.zeros(batch_size, 4)
    dummy_action_encoded.scatter_(1, dummy_action.unsqueeze(1), 1)
    
    print(f"\n🚀 Dynamics Forward Pass実行中...")
    next_latent, reward = model.dynamics_network(latent_state, dummy_action_encoded)
    
    print(f"   - Next latent: {next_latent.shape}")
    print(f"   - Reward: {reward.shape}")
    
    # Loss
    target_latent = torch.randn_like(next_latent)
    target_reward = torch.randn_like(reward)
    loss = ((next_latent - target_latent) ** 2).mean() + ((reward - target_reward) ** 2).mean()
    
    print(f"\n📉 Loss: {loss.item():.6f}")
    
    # Backward
    print(f"\n⬅️  Backward Pass実行中...")
    loss.backward()
    
    # 勾配をチェック
    dynamics_grads = []
    for param, name in zip(dynamics_gnn_params, dynamics_param_names):
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            dynamics_grads.append((name, grad_norm))
    
    print(f"\n✅ Dynamics Network勾配:")
    print(f"   - 勾配あり: {len(dynamics_grads)}/{len(dynamics_gnn_params)}")
    
    if dynamics_grads:
        print(f"\n📈 Dynamics GNN勾配ノルム (上位10個):")
        for name, norm in sorted(dynamics_grads, key=lambda x: -x[1])[:10]:
            print(f"   - {name}: {norm:.6e}")
    
    passed = len(dynamics_grads) > 0
    
    print("\n" + "-"*80)
    if passed:
        print(f"✅ Dynamics Network検証: 合格")
        print(f"   - Dynamics NetworkのGNNに勾配が流れました")
    else:
        print(f"❌ Dynamics Network検証: 失敗")
        print(f"   - Dynamics NetworkのGNNに勾配が流れていません")
    print("-"*80)
    
    return {
        "status": "PASSED" if passed else "FAILED",
        "dynamics_gnn_params": len(dynamics_gnn_params),
        "dynamics_params_with_grad": len(dynamics_grads),
        "loss": loss.item(),
    }


def main():
    """
    メイン検証ロジック
    """
    print("\n" + "="*80)
    print("🔬 GNN Backward Pass (学習フェーズ) 詳細検証")
    print("="*80)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Load config
    print("📦 モジュールをインポート中...")
    from zoo.game_2048.config.stochastic_muzero_2048_gnn_config import main_config, create_config
    from lzero.model import GNNStochasticMuZeroModel
    
    print("✓ インポート完了\n")
    
    # Create model
    print("🏗️  モデルを構築中...")
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
    
    # Run tests
    results = {}
    
    # Test 1: Single backward pass
    results['test1'] = test_single_backward_pass(model, batch_size=8)
    
    # Test 2: Multi-step training
    results['test2'] = test_multi_step_training(model, num_steps=10, batch_size=8)
    
    # Test 3: Dynamics network backward
    results['test3'] = test_dynamics_network_backward(model, batch_size=8)
    
    # Summary
    print("\n" + "="*80)
    print("📊 総合結果サマリー")
    print("="*80)
    
    all_tests = [
        ("単一Backward Pass", results['test1']['status']),
        ("学習シミュレーション", results['test2']['status']),
        ("Dynamics Network Backward", results['test3']['status']),
    ]
    
    print("\nテスト結果:")
    for test_name, status in all_tests:
        symbol = "✅" if status == "PASSED" else ("⚠️" if status == "SKIPPED" else "❌")
        print(f"   {symbol} {test_name}: {status}")
    
    all_passed = all(status == "PASSED" for _, status in all_tests if status != "SKIPPED")
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 すべてのBackward Pass検証に合格しました！")
        print("✅ GNNが正しく学習されています")
        print("✅ GNNパラメータに勾配が流れ、更新されています")
        print("✅ CNNは使用されていません")
    else:
        print("⚠️  一部の検証が失敗しました")
    print("="*80 + "\n")
    
    # 詳細結果を保存
    output = {
        "timestamp": datetime.now().isoformat(),
        "test_results": results,
        "summary": {
            "all_passed": all_passed,
            "tests": [{"name": name, "status": status} for name, status in all_tests],
        }
    }
    
    output_file = "gnn_backward_verification.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"📁 詳細結果を {output_file} に保存しました\n")


if __name__ == "__main__":
    main()
