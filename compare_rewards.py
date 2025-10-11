"""
Compare reward metrics between GNN and CNN training logs
"""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import glob

def load_tb_events(log_dir):
    """Load TensorBoard events from a directory"""
    event_files = glob.glob(os.path.join(log_dir, "**/*.tfevents.*"), recursive=True)
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None
    
    event_file = event_files[0]
    print(f"Loading: {event_file}")
    
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    return ea

def print_scalar_summaries(ea, limit=20):
    """Print scalar summaries"""
    print("\nAvailable scalar tags:")
    tags = ea.Tags()['scalars']
    for tag in tags:
        print(f"  - {tag}")
    
    print("\n" + "="*80)
    
    # Look for reward-related metrics
    reward_tags = [tag for tag in tags if 'reward' in tag.lower() or 'return' in tag.lower()]
    
    for tag in reward_tags:
        events = ea.Scalars(tag)
        print(f"\n{tag}:")
        print(f"  Total events: {len(events)}")
        print(f"  First {min(limit, len(events))} values:")
        for i, event in enumerate(events[:limit]):
            print(f"    Step {event.step}: {event.value:.2f}")

# GNN log
print("="*80)
print("GNN Training Log")
print("="*80)
gnn_log_dir = "/opendilab/2048GNN/LightZero/zoo/game_2048/config/data_gnn_stochastic_mz/gnn_simple_success1/log/serial"
gnn_ea = load_tb_events(gnn_log_dir)
if gnn_ea:
    print_scalar_summaries(gnn_ea, limit=10)

print("\n\n")

# CNN log
print("="*80)
print("CNN Training Log")
print("="*80)
cnn_log_dir = "/opendilab/2048GNN/game_2048_npct-2_stochastic_muzero_ns100_upc200_rer0.0_bs512_chance-True_sslw2_seed0_250729_140944/log/serial"
cnn_ea = load_tb_events(cnn_log_dir)
if cnn_ea:
    print_scalar_summaries(cnn_ea, limit=10)
