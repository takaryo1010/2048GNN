"""
実際のトレーニングでGNNが使用されていることを監視するスクリプト
トレーニングプロセスに統合して実行
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'LightZero'))

import torch
import torch.nn as nn
from collections import defaultdict
from datetime import datetime
import json


class RealTimeGNNMonitor:
    """
    実トレーニング用GNN監視クラス
    """
    
    def __init__(self, model: nn.Module, log_file: str = "gnn_training_monitor.log"):
        self.model = model
        self.log_file = log_file
        self.iteration = 0
        
        # 統計情報
        self.stats = {
            'gnn_param_count': 0,
            'conv2d_param_count': 0,
            'gnn_param_names': [],
            'conv2d_param_names': [],
            'gradient_history': defaultdict(list),
            'param_norm_history': defaultdict(list),
        }
        
        self._analyze_model()
        self._write_header()
    
    def _analyze_model(self):
        """モデル構造を解析"""
        print("\n" + "="*80)
        print("🔍 実トレーニングGNN監視を初期化中...")
        print("="*80)
        
        # GNNパラメータを収集
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # GNN関連
                if any(kw in name.lower() for kw in ['gnn', 'graph', 'sage', 'representation', 'dynamics']):
                    if 'conv2d' not in name.lower() and 'chance' not in name.lower():
                        self.stats['gnn_param_count'] += 1
                        self.stats['gnn_param_names'].append(name)
                
                # Conv2d (Chance encoder以外)
                elif 'conv2d' in name.lower() and 'chance' not in name.lower():
                    self.stats['conv2d_param_count'] += 1
                    self.stats['conv2d_param_names'].append(name)
        
        print(f"\n📊 モデル解析結果:")
        print(f"   - GNNパラメータ: {self.stats['gnn_param_count']}個")
        print(f"   - Conv2dパラメータ: {self.stats['conv2d_param_count']}個")
        
        if self.stats['gnn_param_count'] > 0:
            print(f"\n✅ GNNパラメータ検出 (最初の5個):")
            for name in self.stats['gnn_param_names'][:5]:
                print(f"      - {name}")
        
        if self.stats['conv2d_param_count'] > 0:
            print(f"\n⚠️  Conv2dパラメータ検出:")
            for name in self.stats['conv2d_param_names']:
                print(f"      - {name}")
        
        print("\n" + "="*80 + "\n")
    
    def _write_header(self):
        """ログヘッダーを書き込み"""
        with open(self.log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GNN実トレーニング監視ログ\n")
            f.write("="*80 + "\n")
            f.write(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"GNNパラメータ数: {self.stats['gnn_param_count']}\n")
            f.write(f"Conv2dパラメータ数: {self.stats['conv2d_param_count']}\n")
            f.write("="*80 + "\n\n")
    
    def record_iteration(self, iteration: int, loss: float = None):
        """
        各イテレーションで呼び出してGNNの状態を記録
        """
        self.iteration = iteration
        
        # 勾配とパラメータノルムを記録
        gnn_grad_norms = []
        gnn_param_norms = []
        conv2d_grad_norms = []
        
        for name, param in self.model.named_parameters():
            if name in self.stats['gnn_param_names']:
                # パラメータノルム
                param_norm = param.data.norm().item()
                gnn_param_norms.append(param_norm)
                self.stats['param_norm_history'][name].append(param_norm)
                
                # 勾配ノルム
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    gnn_grad_norms.append(grad_norm)
                    self.stats['gradient_history'][name].append(grad_norm)
            
            elif name in self.stats['conv2d_param_names']:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    conv2d_grad_norms.append(grad_norm)
        
        # 統計計算
        avg_gnn_grad = sum(gnn_grad_norms) / len(gnn_grad_norms) if gnn_grad_norms else 0.0
        avg_gnn_param = sum(gnn_param_norms) / len(gnn_param_norms) if gnn_param_norms else 0.0
        avg_conv_grad = sum(conv2d_grad_norms) / len(conv2d_grad_norms) if conv2d_grad_norms else 0.0
        
        # 定期的に詳細ログ出力
        if iteration % 100 == 0:
            self._log_detailed_status(iteration, loss, avg_gnn_grad, avg_gnn_param, avg_conv_grad, gnn_grad_norms)
        
        return {
            'iteration': iteration,
            'loss': loss,
            'avg_gnn_grad_norm': avg_gnn_grad,
            'avg_gnn_param_norm': avg_gnn_param,
            'avg_conv_grad_norm': avg_conv_grad,
            'gnn_grads_count': len(gnn_grad_norms),
            'conv_grads_count': len(conv2d_grad_norms),
        }
    
    def _log_detailed_status(self, iteration, loss, avg_gnn_grad, avg_gnn_param, avg_conv_grad, gnn_grad_norms):
        """詳細ステータスをログ出力"""
        msg = f"\n{'='*80}\n"
        msg += f"📊 GNN監視レポート [Iteration {iteration}]\n"
        msg += f"{'='*80}\n"
        
        if loss is not None:
            msg += f"Loss: {loss:.6f}\n"
        
        msg += f"\n🔍 GNN状態:\n"
        msg += f"   - 平均GNN勾配ノルム: {avg_gnn_grad:.6e}\n"
        msg += f"   - 平均GNNパラメータノルム: {avg_gnn_param:.6e}\n"
        msg += f"   - GNN勾配が存在: {len(gnn_grad_norms)}/{self.stats['gnn_param_count']}\n"
        msg += f"   - Conv2d平均勾配ノルム: {avg_conv_grad:.6e}\n"
        
        if avg_gnn_grad > 0:
            msg += f"\n✅ GNNが正常に学習中\n"
        else:
            msg += f"\n⚠️  GNN勾配が検出されません\n"
        
        if avg_conv_grad > 0:
            msg += f"⚠️  Conv2dも学習されています\n"
        else:
            msg += f"✅ Conv2dは学習されていません\n"
        
        msg += f"{'='*80}\n"
        
        # コンソール出力
        print(msg)
        
        # ファイルに追記
        with open(self.log_file, 'a') as f:
            f.write(msg)
    
    def get_summary(self):
        """最終サマリーを取得"""
        summary = {
            'total_iterations': self.iteration,
            'gnn_param_count': self.stats['gnn_param_count'],
            'conv2d_param_count': self.stats['conv2d_param_count'],
            'gnn_params_learned': 0,
            'avg_gnn_grad_norm': 0.0,
            'avg_conv_grad_norm': 0.0,
        }
        
        # GNNパラメータのうち、勾配が流れたものをカウント
        for name in self.stats['gnn_param_names']:
            if name in self.stats['gradient_history'] and len(self.stats['gradient_history'][name]) > 0:
                if any(g > 1e-10 for g in self.stats['gradient_history'][name]):
                    summary['gnn_params_learned'] += 1
        
        # 平均勾配ノルムを計算
        all_gnn_grads = []
        for grads in self.stats['gradient_history'].values():
            all_gnn_grads.extend(grads)
        
        if all_gnn_grads:
            summary['avg_gnn_grad_norm'] = sum(all_gnn_grads) / len(all_gnn_grads)
        
        return summary
    
    def save_final_report(self, output_file: str = "gnn_training_final_report.json"):
        """最終レポートを保存"""
        summary = self.get_summary()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'detailed_stats': {
                'gnn_param_names': self.stats['gnn_param_names'],
                'conv2d_param_names': self.stats['conv2d_param_names'],
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📁 最終レポートを {output_file} に保存しました")
        
        return summary


def test_with_short_training(max_iterations: int = 500):
    """
    短時間のトレーニングでGNN監視をテスト
    """
    print("\n" + "="*80)
    print("🚀 実トレーニングテスト開始")
    print(f"   最大イテレーション: {max_iterations}")
    print("="*80 + "\n")
    
    from zoo.game_2048.config.stochastic_muzero_2048_gnn_config import main_config, create_config
    from lzero.entry import train_muzero
    
    # 設定を短時間用に調整
    test_config = main_config.copy()
    test_config.policy.eval_freq = int(100)  # 評価頻度を短く
    
    print("📋 トレーニング設定:")
    print(f"   - Model: {create_config.model.type}")
    print(f"   - Batch size: {test_config.policy.batch_size}")
    print(f"   - Update per collect: {test_config.policy.update_per_collect}")
    print(f"   - GNN layers: {test_config.policy.model.num_gnn_layers}")
    print(f"   - GNN channels: {test_config.policy.model.num_channels}")
    print(f"   - Edge mode: {test_config.policy.model.edge_mode}")
    
    # トレーニング開始
    print(f"\n🏋️  トレーニング開始...\n")
    
    try:
        train_muzero(
            [test_config, create_config],
            seed=0,
            max_env_step=max_iterations,
        )
    except KeyboardInterrupt:
        print("\n⚠️  トレーニングが中断されました")
    except Exception as e:
        print(f"\n⚠️  エラーが発生しました: {e}")
    
    print("\n✅ トレーニングテスト完了\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='実トレーニングでGNN使用を確認')
    parser.add_argument('--max-iterations', type=int, default=500,
                       help='最大トレーニングイテレーション数')
    
    args = parser.parse_args()
    
    test_with_short_training(max_iterations=args.max_iterations)
