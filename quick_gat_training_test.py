"""
Quick training test for GAT-based Stochastic MuZero
Tests a few iterations to verify training pipeline works
"""
import sys
sys.path.insert(0, '/opendilab/2048GNN/LightZero')

def test_gat_training():
    """Test GAT model training for a few iterations"""
    print("="*80)
    print("Quick GAT Training Test")
    print("="*80)
    
    # Import after path setup
    from lzero.entry import train_muzero
    from zoo.game_2048.config.stochastic_muzero_2048_gat_config import (
        main_config, create_config
    )
    import torch
    
    # Check CUDA
    if torch.cuda.is_available():
        print("\n✅ CUDA is available. Training on GPU.")
        device = "cuda"
    else:
        print("\n⚠️  CUDA is not available. Training on CPU (slower).")
        device = "cpu"
    
    # Modify config for quick test
    main_config.policy.cuda = torch.cuda.is_available()
    main_config.policy.collector_env_num = 2
    main_config.policy.evaluator_env_num = 1
    main_config.policy.n_episode = 4  # Must be >= collector_env_num
    main_config.policy.batch_size = 64
    main_config.policy.update_per_collect = 10
    main_config.policy.num_simulations = 10
    main_config.env.collector_env_num = 2
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    
    # Very short test
    max_env_step = 500  # Just 500 steps
    
    print(f"\nConfiguration:")
    print(f"  - Model type: GAT (Graph Attention Network)")
    print(f"  - Attention heads: {main_config.policy.model.num_heads}")
    print(f"  - GAT layers: {main_config.policy.model.num_gnn_layers}")
    print(f"  - Hidden dim: {main_config.policy.model.num_channels}")
    print(f"  - Edge mode: {main_config.policy.model.edge_mode}")
    print(f"  - Batch size: {main_config.policy.batch_size}")
    print(f"  - Max env steps: {max_env_step}")
    print(f"  - Device: {device}")
    
    print("\n" + "="*80)
    print("Starting training test (this will take a few moments)...")
    print("="*80 + "\n")
    
    try:
        train_muzero(
            [main_config, create_config],
            seed=0,
            model_path=None,
            max_env_step=max_env_step
        )
        print("\n" + "="*80)
        print("✅ Training test completed successfully!")
        print("="*80)
        return True
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ Training test failed: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gat_training()
    sys.exit(0 if success else 1)
