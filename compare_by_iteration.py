"""
Compare by iteration (training iteration) instead of env steps
"""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

def compare_by_iteration(gnn_dir, cnn_dir):
    """Compare metrics by iteration number"""
    
    # Load GNN
    gnn_files = glob.glob(f"{gnn_dir}/**/*.tfevents.*", recursive=True)
    gnn_ea = EventAccumulator(gnn_files[0])
    gnn_ea.Reload()
    
    # Load CNN
    cnn_files = glob.glob(f"{cnn_dir}/**/*.tfevents.*", recursive=True)
    cnn_ea = EventAccumulator(cnn_files[0])
    cnn_ea.Reload()
    
    print("="*80)
    print("Comparing by ITERATION (not env steps)")
    print("="*80)
    
    # Use evaluator_iter metrics (iteration-based, not step-based)
    metrics = [
        'evaluator_iter/eval_episode_return_mean',
        'evaluator_iter/reward_mean',
    ]
    
    for metric in metrics:
        print(f"\n{metric}:")
        print(f"{'Iter':>6} | {'GNN':>12} | {'CNN':>12} | {'Ratio (GNN/CNN)':>15}")
        print("-" * 80)
        
        gnn_data = gnn_ea.Scalars(metric)
        cnn_data = cnn_ea.Scalars(metric)
        
        # Compare first 30 iterations
        max_len = min(30, len(gnn_data), len(cnn_data))
        for i in range(max_len):
            gnn_val = gnn_data[i].value
            cnn_val = cnn_data[i].value
            iter_num = gnn_data[i].step
            ratio = gnn_val / cnn_val if cnn_val != 0 else 0
            print(f"{iter_num:>6} | {gnn_val:>12.2f} | {cnn_val:>12.2f} | {ratio:>15.3f}")

gnn_log_dir = "/opendilab/2048GNN/LightZero/zoo/game_2048/config/data_gnn_stochastic_mz/gnn_simple_success1/log/serial"
cnn_log_dir = "/opendilab/2048GNN/game_2048_npct-2_stochastic_muzero_ns100_upc200_rer0.0_bs512_chance-True_sslw2_seed0_250729_140944/log/serial"

compare_by_iteration(gnn_log_dir, cnn_log_dir)
