"""
Detailed comparison focusing on 80k steps
"""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

def analyze_around_80k(gnn_dir, cnn_dir):
    """Analyze metrics around 80k steps"""
    
    # Load GNN
    gnn_files = glob.glob(f"{gnn_dir}/**/*.tfevents.*", recursive=True)
    gnn_ea = EventAccumulator(gnn_files[0])
    gnn_ea.Reload()
    
    # Load CNN
    cnn_files = glob.glob(f"{cnn_dir}/**/*.tfevents.*", recursive=True)
    cnn_ea = EventAccumulator(cnn_files[0])
    cnn_ea.Reload()
    
    # Find values around 80k steps
    target_step = 80000
    margin = 10000
    
    print("="*80)
    print(f"Metrics around {target_step} steps (±{margin})")
    print("="*80)
    
    # Check multiple metrics
    metrics = [
        'evaluator_step/eval_episode_return_mean',
        'evaluator_step/reward_mean',
        'collector_step/reward_mean',
    ]
    
    for metric in metrics:
        print(f"\n{metric}:")
        print(f"{'':>5} | {'Step':>10} | {'GNN':>12} | {'CNN':>12} | {'Ratio':>8}")
        print("-" * 80)
        
        gnn_data = gnn_ea.Scalars(metric)
        cnn_data = cnn_ea.Scalars(metric)
        
        # Find entries near target
        gnn_near = [(i, e) for i, e in enumerate(gnn_data) 
                    if target_step - margin <= e.step <= target_step + margin]
        cnn_near = [(i, e) for i, e in enumerate(cnn_data) 
                    if target_step - margin <= e.step <= target_step + margin]
        
        # Match by index
        max_len = min(len(gnn_near), len(cnn_near))
        for i in range(max_len):
            gnn_idx, gnn_event = gnn_near[i]
            cnn_idx, cnn_event = cnn_near[i]
            ratio = gnn_event.value / cnn_event.value if cnn_event.value != 0 else 0
            print(f"GNN | {gnn_event.step:>10} | {gnn_event.value:>12.2f} | {'':<12} | {'':<8}")
            print(f"CNN | {cnn_event.step:>10} | {'':<12} | {cnn_event.value:>12.2f} | {ratio:>8.3f}")
            print()

gnn_log_dir = "/opendilab/2048GNN/LightZero/zoo/game_2048/config/data_gnn_stochastic_mz/gnn_simple_success1/log/serial"
cnn_log_dir = "/opendilab/2048GNN/game_2048_npct-2_stochastic_muzero_ns100_upc200_rer0.0_bs512_chance-True_sslw2_seed0_250729_140944/log/serial"

analyze_around_80k(gnn_log_dir, cnn_log_dir)
