"""
トレーニング中にGNNの使用状況をリアルタイム監視するスクリプト
- GNN層の勾配フローを確認
- GNN層のアクティベーションを監視
- CNNレイヤーが使用されていないことを確認
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'LightZero'))

import torch
import torch.nn as nn
from collections import defaultdict
import time
from typing import Dict, List


class GNNTrainingMonitor:
    """
    GNNトレーニングのモニタークラス
    """
    
    def __init__(self, model: nn.Module, log_interval: int = 100):
        """
        Args:
            model: 監視対象のモデル
            log_interval: ログ出力間隔（イテレーション数）
        """
        self.model = model
        self.log_interval = log_interval
        self.iteration = 0
        
        # 統計情報
        self.stats = {
            'gnn_forward_count': 0,
            'gnn_backward_count': 0,
            'conv2d_forward_count': 0,
            'conv2d_backward_count': 0,
            'gnn_layers': [],
            'conv2d_layers': [],
            'gnn_gradient_norms': defaultdict(list),
            'gnn_activation_norms': defaultdict(list),
        }
        
        self.hooks = []
        self._setup_hooks()
    
    def _setup_hooks(self):
        """
        Forward/Backwardフックをセットアップ
        """
        print("\n🔧 GNN監視フックを設定中...")
        
        for name, module in self.model.named_modules():
            # GNN関連レイヤー
            if any(kw in type(module).__name__ for kw in ['GraphSAGE', 'GNN', 'GraphBuilder', 'GAT']):
                self.stats['gnn_layers'].append(name)
                
                # Forward hook
                hook = module.register_forward_hook(
                    self._make_forward_hook(name, 'GNN')
                )
                self.hooks.append(hook)
                
                # Backward hook (勾配を監視)
                if hasattr(module, 'weight') and module.weight is not None:
                    hook = module.weight.register_hook(
                        self._make_grad_hook(name, 'GNN')
                    )
                    self.hooks.append(hook)
            
            # Conv2dレイヤー（使用されていないことを確認）
            elif isinstance(module, nn.Conv2d):
                # Chance encoder以外のConv2dを監視
                if 'chance' not in name.lower():
                    self.stats['conv2d_layers'].append(name)
                    
                    # Forward hook
                    hook = module.register_forward_hook(
                        self._make_forward_hook(name, 'Conv2d')
                    )
                    self.hooks.append(hook)
        
        print(f"   ✓ GNNレイヤー: {len(self.stats['gnn_layers'])}個")
        print(f"   ✓ Conv2dレイヤー: {len(self.stats['conv2d_layers'])}個")
        print("   ✓ フックのセットアップ完了\n")
    
    def _make_forward_hook(self, name: str, layer_type: str):
        """Forward hookを作成"""
        def hook(module, input, output):
            if layer_type == 'GNN':
                self.stats['gnn_forward_count'] += 1
                
                # Activation norm
                if isinstance(output, torch.Tensor):
                    norm = output.norm().item()
                    self.stats['gnn_activation_norms'][name].append(norm)
            
            elif layer_type == 'Conv2d':
                self.stats['conv2d_forward_count'] += 1
                print(f"⚠️  WARNING: Conv2d layer '{name}' was activated!")
        
        return hook
    
    def _make_grad_hook(self, name: str, layer_type: str):
        """Gradient hookを作成"""
        def hook(grad):
            if layer_type == 'GNN':
                self.stats['gnn_backward_count'] += 1
                
                # Gradient norm
                if grad is not None:
                    norm = grad.norm().item()
                    self.stats['gnn_gradient_norms'][name].append(norm)
        
        return hook
    
    def step(self):
        """
        各トレーニングステップで呼び出す
        """
        self.iteration += 1
        
        if self.iteration % self.log_interval == 0:
            self._log_stats()
    
    def _log_stats(self):
        """
        統計情報をログ出力
        """
        print("\n" + "="*80)
        print(f"📊 GNN監視レポート [Iteration {self.iteration}]")
        print("="*80)
        
        print(f"\n🔄 Forward/Backward カウント:")
        print(f"   - GNN Forward: {self.stats['gnn_forward_count']}")
        print(f"   - GNN Backward: {self.stats['gnn_backward_count']}")
        print(f"   - Conv2d Forward: {self.stats['conv2d_forward_count']}")
        print(f"   - Conv2d Backward: {self.stats['conv2d_backward_count']}")
        
        # GNN層の勾配ノルム
        if self.stats['gnn_gradient_norms']:
            print(f"\n📈 GNN層の勾配ノルム (最近の値):")
            for name, norms in list(self.stats['gnn_gradient_norms'].items())[:3]:
                if norms:
                    avg_norm = sum(norms[-10:]) / len(norms[-10:])
                    print(f"   - {name}: {avg_norm:.6f}")
        
        # GNN層のアクティベーションノルム
        if self.stats['gnn_activation_norms']:
            print(f"\n🎯 GNN層のアクティベーションノルム (最近の値):")
            for name, norms in list(self.stats['gnn_activation_norms'].items())[:3]:
                if norms:
                    avg_norm = sum(norms[-10:]) / len(norms[-10:])
                    print(f"   - {name}: {avg_norm:.6f}")
        
        # 検証結果
        print(f"\n✅ 検証結果:")
        if self.stats['gnn_forward_count'] > 0:
            print(f"   ✓ GNNレイヤーが正しく動作しています")
        else:
            print(f"   ✗ GNNレイヤーが動作していません！")
        
        if self.stats['conv2d_forward_count'] == 0:
            print(f"   ✓ Conv2dレイヤーは使用されていません（CNN不使用）")
        else:
            print(f"   ⚠️  Conv2dレイヤーが{self.stats['conv2d_forward_count']}回動作しました")
        
        print("="*80 + "\n")
    
    def cleanup(self):
        """
        フックをクリーンアップ
        """
        for hook in self.hooks:
            hook.remove()
        print("🧹 監視フックをクリーンアップしました")
    
    def get_summary(self) -> Dict:
        """
        最終サマリーを取得
        """
        return {
            "total_iterations": self.iteration,
            "gnn_forward_count": self.stats['gnn_forward_count'],
            "gnn_backward_count": self.stats['gnn_backward_count'],
            "conv2d_forward_count": self.stats['conv2d_forward_count'],
            "conv2d_backward_count": self.stats['conv2d_backward_count'],
            "num_gnn_layers": len(self.stats['gnn_layers']),
            "num_conv2d_layers": len(self.stats['conv2d_layers']),
            "gnn_layers": self.stats['gnn_layers'],
            "conv2d_layers": self.stats['conv2d_layers'],
        }


def test_monitor_with_dummy_training():
    """
    ダミートレーニングでモニターをテスト
    """
    print("\n🧪 ダミートレーニングでGNN監視をテスト中...\n")
    
    # Load config
    from zoo.game_2048.config.stochastic_muzero_2048_gnn_config import main_config, create_config
    
    # Import model
    from lzero.model import GNNStochasticMuZeroModel
    
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
    
    print("✓ モデルを構築しました")
    
    # Setup monitor
    monitor = GNNTrainingMonitor(model, log_interval=5)
    
    # Dummy optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n🏃 ダミートレーニング開始...\n")
    
    # Dummy training loop
    num_iterations = 20
    batch_size = 8
    
    try:
        for i in range(num_iterations):
            # Forward pass
            dummy_obs = torch.randn(batch_size, 16, 4, 4)
            
            # Representation network
            latent_state = model.representation_network(dummy_obs)
            
            # Prediction network
            value, policy_logits = model.prediction_network(latent_state)
            
            # Dummy loss
            target_value = torch.randn_like(value)
            target_policy = torch.randn_like(policy_logits)
            
            loss = ((value - target_value) ** 2).mean() + \
                   ((policy_logits - target_policy) ** 2).mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Monitor step
            monitor.step()
            
            if (i + 1) % 5 == 0:
                print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}")
    
    finally:
        # Cleanup
        monitor.cleanup()
    
    # Final summary
    summary = monitor.get_summary()
    
    print("\n" + "="*80)
    print("📋 最終サマリー")
    print("="*80)
    print(f"総イテレーション数: {summary['total_iterations']}")
    print(f"GNN Forward回数: {summary['gnn_forward_count']}")
    print(f"GNN Backward回数: {summary['gnn_backward_count']}")
    print(f"Conv2d Forward回数: {summary['conv2d_forward_count']}")
    print(f"Conv2d Backward回数: {summary['conv2d_backward_count']}")
    print(f"\nGNN層数: {summary['num_gnn_layers']}")
    print(f"Conv2d層数: {summary['num_conv2d_layers']}")
    
    print("\n✅ 結論:")
    if summary['gnn_forward_count'] > 0 and summary['conv2d_forward_count'] == 0:
        print("   GNNが正しく使用されており、CNNは使用されていません！")
    elif summary['gnn_forward_count'] > 0 and summary['conv2d_forward_count'] > 0:
        print("   ⚠️  GNNとCNNの両方が使用されています")
    elif summary['gnn_forward_count'] == 0:
        print("   ✗ GNNが使用されていません！")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    test_monitor_with_dummy_training()
