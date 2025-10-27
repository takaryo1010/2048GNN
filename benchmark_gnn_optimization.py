"""
GNN最適化版と既存版の性能比較スクリプト
"""
import torch
import time
import numpy as np
from lzero.model.gnn_stochastic_muzero_model import GNNStochasticMuZeroModel
from lzero.model.gnn_stochastic_muzero_model_optimized import GNNStochasticMuZeroModelOptimized

def benchmark_model(model, name, num_iterations=100, batch_size=64):
    """モデルのベンチマーク"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    # ダミーデータ
    obs = torch.randn(batch_size, 16, 4, 4)
    action = torch.randint(0, 4, (batch_size,))
    
    if torch.cuda.is_available():
        model = model.cuda()
        obs = obs.cuda()
        action = action.cuda()
        device_name = torch.cuda.get_device_name(0)
        print(f"Device: {device_name}")
    else:
        print("Device: CPU")
    
    model.eval()
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            output = model.initial_inference(obs)
            _ = model.recurrent_inference(output.latent_state, action, afterstate=False)
    
    # Benchmark - Initial Inference
    print("\nBenchmarking initial_inference...")
    times_initial = []
    for _ in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            output = model.initial_inference(obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_initial.append(time.time() - start)
    
    # Benchmark - Recurrent Inference
    print("Benchmarking recurrent_inference...")
    output = model.initial_inference(obs)
    times_recurrent = []
    for _ in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model.recurrent_inference(output.latent_state, action, afterstate=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_recurrent.append(time.time() - start)
    
    # 結果
    print(f"\n{'='*60}")
    print(f"Results for {name}")
    print(f"{'='*60}")
    print(f"Initial Inference:")
    print(f"  Mean: {np.mean(times_initial)*1000:.3f} ms")
    print(f"  Std:  {np.std(times_initial)*1000:.3f} ms")
    print(f"\nRecurrent Inference:")
    print(f"  Mean: {np.mean(times_recurrent)*1000:.3f} ms")
    print(f"  Std:  {np.std(times_recurrent)*1000:.3f} ms")
    
    # メモリ使用量
    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  Cached:    {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    return {
        'initial_mean': np.mean(times_initial),
        'initial_std': np.std(times_initial),
        'recurrent_mean': np.mean(times_recurrent),
        'recurrent_std': np.std(times_recurrent),
    }

def main():
    print("="*60)
    print("GNN Model Performance Comparison")
    print("="*60)
    
    # モデル設定
    config = {
        'observation_shape': (16, 4, 4),
        'action_space_size': 4,
        'chance_space_size': 32,
        'num_channels': 128,
        'num_gnn_layers': 3,
        'grid_size': 4,
        'value_head_hidden_channels': [128, 64],
        'policy_head_hidden_channels': [128, 64],
        'reward_head_hidden_channels': [128, 64],
        'reward_support_size': 601,
        'value_support_size': 601,
        'categorical_distribution': True,
        'edge_mode': 'sparse',
    }
    
    # 既存版
    print("\n1. Creating Base GNN Model...")
    model_base = GNNStochasticMuZeroModel(**config)
    print(f"   Parameters: {sum(p.numel() for p in model_base.parameters()):,}")
    
    # 最適化版
    print("\n2. Creating Optimized GNN Model...")
    model_opt = GNNStochasticMuZeroModelOptimized(**config)
    print(f"   Parameters: {sum(p.numel() for p in model_opt.parameters()):,}")
    
    # ベンチマーク
    results_base = benchmark_model(model_base, "Base GNN Model", num_iterations=100)
    
    # GPU メモリをクリア
    if torch.cuda.is_available():
        del model_base
        torch.cuda.empty_cache()
    
    results_opt = benchmark_model(model_opt, "Optimized GNN Model", num_iterations=100)
    
    # 比較
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    speedup_initial = (results_base['initial_mean'] / results_opt['initial_mean'] - 1) * 100
    speedup_recurrent = (results_base['recurrent_mean'] / results_opt['recurrent_mean'] - 1) * 100
    
    print(f"\nInitial Inference:")
    print(f"  Base:      {results_base['initial_mean']*1000:.3f} ms")
    print(f"  Optimized: {results_opt['initial_mean']*1000:.3f} ms")
    print(f"  Speedup:   {speedup_initial:+.1f}%")
    
    print(f"\nRecurrent Inference:")
    print(f"  Base:      {results_base['recurrent_mean']*1000:.3f} ms")
    print(f"  Optimized: {results_opt['recurrent_mean']*1000:.3f} ms")
    print(f"  Speedup:   {speedup_recurrent:+.1f}%")
    
    print(f"\n{'='*60}")
    if speedup_initial > 0 and speedup_recurrent > 0:
        print("✅ Optimized version is FASTER!")
        print(f"   Average speedup: {(speedup_initial + speedup_recurrent)/2:.1f}%")
    else:
        print("⚠️  Optimization results may vary by hardware")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
