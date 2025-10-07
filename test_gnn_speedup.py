"""
Test GNN model speed with different optimization settings
"""
import torch
import time
import sys
sys.path.append('/opendilab/2048GNN/LightZero')

from lzero.model.gnn_stochastic_muzero_model import GNNRepresentationNetwork

def benchmark_model(edge_mode, num_batches=50, batch_size=512):
    """Benchmark a model configuration"""
    print(f"\n{'='*60}")
    print(f"Testing edge_mode='{edge_mode}'")
    print(f"{'='*60}")
    
    # Create model
    model = GNNRepresentationNetwork(
        observation_shape=(16, 4, 4),
        num_channels=128,
        num_gnn_layers=3,
        grid_size=4,
        include_row_col_edges=True,
        dropout=0.0,
        edge_mode=edge_mode,
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Count edges
    edge_count = model.graph_builder.edge_index.size(1)
    print(f"Number of edges: {edge_count}")
    
    # Warmup
    dummy_input = torch.randn(batch_size, 16, 4, 4).to(device)
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    for i in range(num_batches):
        dummy_input = torch.randn(batch_size, 16, 4, 4).to(device)
        
        start = time.perf_counter()
        with torch.no_grad():
            output = model(dummy_input)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
        
        if (i + 1) % 10 == 0:
            avg_time = sum(times) / len(times)
            print(f"Batch {i+1}/{num_batches}: avg {avg_time*1000:.2f}ms per batch")
    
    avg_time = sum(times) / len(times)
    throughput = batch_size / avg_time
    
    print(f"\nResults for edge_mode='{edge_mode}':")
    print(f"  Average time: {avg_time*1000:.2f} ms/batch")
    print(f"  Throughput: {throughput:.0f} samples/sec")
    print(f"  Total edges: {edge_count}")
    
    return avg_time, edge_count

if __name__ == "__main__":
    print("GNN Speed Optimization Benchmark")
    print(f"Batch size: 512")
    print(f"Grid size: 4x4")
    print(f"GNN layers: 3")
    print(f"Hidden dim: 128")
    
    results = {}
    
    # Test each edge mode
    for mode in ['adjacent', 'sparse', 'full']:
        try:
            avg_time, edge_count = benchmark_model(mode, num_batches=50)
            results[mode] = {'time': avg_time, 'edges': edge_count}
        except Exception as e:
            print(f"Error testing {mode}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mode':<12} {'Edges':<10} {'Time (ms)':<15} {'Speedup':<10}")
    print(f"{'-'*60}")
    
    baseline = results.get('full', {}).get('time', 1)
    for mode in ['adjacent', 'sparse', 'full']:
        if mode in results:
            time_ms = results[mode]['time'] * 1000
            speedup = baseline / results[mode]['time']
            edges = results[mode]['edges']
            print(f"{mode:<12} {edges:<10} {time_ms:<15.2f} {speedup:<10.2f}x")
    
    print(f"\nRecommendation:")
    print(f"  - For 4x4 grid: Use 'sparse' (best speed/accuracy tradeoff)")
    print(f"  - For 3x3 grid: Use 'adjacent' (small grid, dense connections not needed)")
