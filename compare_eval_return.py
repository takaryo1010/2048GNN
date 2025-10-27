"""
Compare eval_episode_return between GNN and CNN
"""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

def load_and_compare(gnn_dir, cnn_dir):
    """Load and compare eval metrics"""
    
    # Load GNN
    gnn_files = glob.glob(f"{gnn_dir}/**/*.tfevents.*", recursive=True)
    gnn_ea = EventAccumulator(gnn_files[0])
    gnn_ea.Reload()
    
    # Load CNN
    cnn_files = glob.glob(f"{cnn_dir}/**/*.tfevents.*", recursive=True)
    cnn_ea = EventAccumulator(cnn_files[0])
    cnn_ea.Reload()
    
    # Get eval_episode_return_mean
    print("="*80)
    print("Comparing evaluator_step/eval_episode_return_mean (first 20 values)")
    print("="*80)
    
    gnn_eval = gnn_ea.Scalars('evaluator_step/eval_episode_return_mean')
    cnn_eval = cnn_ea.Scalars('evaluator_step/eval_episode_return_mean')
    
    print(f"\n{'Step':>10} | {'GNN Return':>12} | {'CNN Return':>12} | {'Ratio (GNN/CNN)':>15}")
    print("-" * 80)
    
    for i in range(min(20, len(gnn_eval), len(cnn_eval))):
        gnn_val = gnn_eval[i].value
        cnn_val = cnn_eval[i].value
        ratio = gnn_val / cnn_val if cnn_val != 0 else 0
        print(f"{gnn_eval[i].step:>10} | {gnn_val:>12.2f} | {cnn_val:>12.2f} | {ratio:>15.3f}")
    
    # Also compare reward_mean
    print("\n" + "="*80)
    print("Comparing evaluator_step/reward_mean (first 20 values)")
    print("="*80)
    
    gnn_reward = gnn_ea.Scalars('evaluator_step/reward_mean')
    cnn_reward = cnn_ea.Scalars('evaluator_step/reward_mean')
    
    print(f"\n{'Step':>10} | {'GNN Reward':>12} | {'CNN Reward':>12} | {'Ratio (GNN/CNN)':>15}")
    print("-" * 80)
    
    for i in range(min(20, len(gnn_reward), len(cnn_reward))):
        gnn_val = gnn_reward[i].value
        cnn_val = cnn_reward[i].value
        ratio = gnn_val / cnn_val if cnn_val != 0 else 0
        print(f"{gnn_reward[i].step:>10} | {gnn_val:>12.2f} | {cnn_val:>12.2f} | {ratio:>15.3f}")

# Main
gnn_log_dir = "/opendilab/2048GNN/LightZero/zoo/game_2048/config/data_gnn_stochastic_mz/gnn_simple_success1/log/serial"
cnn_log_dir = "/opendilab/2048GNN/game_2048_npct-2_stochastic_muzero_ns100_upc200_rer0.0_bs512_chance-True_sslw2_seed0_250729_140944/log/serial"

load_and_compare(gnn_log_dir, cnn_log_dir)
