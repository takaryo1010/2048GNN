"""
GUI evaluation script for GNN-based Stochastic MuZero on 2048
This script loads a trained GNN model and visualizes gameplay
"""
import numpy as np
import os

from lzero.entry import eval_muzero
# Import the GNN config instead of the standard config
from zoo.game_2048.config.stochastic_muzero_2048_gnn_config import main_config, create_config

if __name__ == "__main__":
    """
    Entry point for GUI evaluation of the GNN-based Stochastic MuZero model on the 2048 environment.
    
    This script:
    1. Loads a trained GNN model checkpoint
    2. Runs the model on 2048 game episodes
    3. Generates visual output (GIF or MP4) showing the gameplay
    
    Variables:
        - model_path (:obj:`Optional[str]`): Path to the pretrained model checkpoint.
          Should point to a .pth.tar file. If None, uses a randomly initialized model.
          Example: './data_gnn_stochastic_mz/game_2048_gnn_*/ckpt/ckpt_best.pth.tar'
        - returns_mean_seeds (:obj:`List[float]`): Mean returns for each seed.
        - returns_seeds (:obj:`List[float]`): All episode returns for each seed.
        - seeds (:obj:`List[int]`): Random seeds for reproducibility.
        - num_episodes_each_seed (:obj:`int`): Number of episodes to run per seed.
        - total_test_episodes (:obj:`int`): Total episodes to evaluate.
    
    Output:
        - Video files (GIF or MP4) saved to the replay_path directory
        - Console output showing max tiles reached and episode rewards
    """
    
    # ========================================
    # CONFIGURATION - Modify these settings
    # ========================================
    
    # Model checkpoint path - set this to your trained model
    # Example: model_path = './data_gnn_stochastic_mz/game_2048_gnn_npct-2_ns100_upc200_rer0.0_bs512_gnn3L128D_sparse_seed0/ckpt/ckpt_best.pth.tar'
    model_path = None  # Set to None to use a randomly initialized model
    
    # Evaluation settings
    seeds = [0]  # Random seeds to test
    num_episodes_each_seed = 3  # Number of episodes per seed
    
    # Rendering settings
    render_mode = 'image_savefile_mode'  # Options: 'image_savefile_mode', 'image_realtime_mode'
    replay_path = './video_gnn'  # Directory to save videos
    replay_format = 'gif'  # Options: 'gif', 'mp4'
    replay_name_suffix = 'gnn_stochastic_muzero_ns100'
    
    # ========================================
    # Environment configuration
    # ========================================
    
    # Enable rendering
    main_config.env.render_mode = render_mode
    main_config.env.replay_path = replay_path
    main_config.env.replay_format = replay_format
    main_config.env.replay_name_suffix = replay_name_suffix
    
    # Set very high max steps to let the game play until naturally done
    main_config.env.max_episode_steps = int(1e9)
    
    # Visualization requires base environment manager and single env
    total_test_episodes = num_episodes_each_seed * len(seeds)
    create_config.env_manager.type = 'base'  # Required for visualization
    main_config.env.evaluator_env_num = 1    # Required for visualization
    main_config.env.n_evaluator_episode = total_test_episodes
    
    # ========================================
    # Run evaluation
    # ========================================
    
    print("=" * 70)
    print("GNN-based Stochastic MuZero - 2048 GUI Evaluation")
    print("=" * 70)
    print(f"Model path: {model_path if model_path else 'Random initialization'}")
    print(f"Seeds: {seeds}")
    print(f"Episodes per seed: {num_episodes_each_seed}")
    print(f"Total episodes: {total_test_episodes}")
    print(f"Render mode: {render_mode}")
    print(f"Output path: {replay_path}")
    print(f"Output format: {replay_format}")
    print("=" * 70)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(replay_path):
        os.makedirs(replay_path)
        print(f"Created output directory: {replay_path}")
    
    returns_mean_seeds = []
    returns_seeds = []
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Evaluating seed {seed}...")
        print(f"{'='*70}")
        
        returns_mean, returns = eval_muzero(
            [main_config, create_config],
            seed=seed,
            num_episodes_each_seed=num_episodes_each_seed,
            print_seed_details=True,
            model_path=model_path
        )
        
        print(f"\nSeed {seed} results:")
        print(f"  Mean return: {returns_mean:.2f}")
        print(f"  Episode returns: {returns}")
        
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)
    
    # ========================================
    # Summary statistics
    # ========================================
    
    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total seeds evaluated: {len(seeds)}")
    print(f"Episodes per seed: {num_episodes_each_seed}")
    print(f"Seeds tested: {seeds}")
    print(f"\nMean returns per seed: {returns_mean_seeds}")
    print(f"All episode returns: {returns_seeds}")
    print(f"\nOverall mean reward: {returns_mean_seeds.mean():.2f}")
    print(f"Overall std reward: {returns_mean_seeds.std():.2f}")
    print(f"Best episode reward: {returns_seeds.max():.2f}")
    print(f"Worst episode reward: {returns_seeds.min():.2f}")
    print("=" * 70)
    print(f"\nVideo files saved to: {replay_path}")
    print("=" * 70)
