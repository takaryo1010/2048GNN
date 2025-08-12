#!/usr/bin/env python3
"""
Test GAT configuration loading and policy creation
"""

import sys
sys.path.insert(0, '/opendilab/LightZero')

print("=== GAT Configuration Test ===")

try:
    # Import LightZero to register all policies first
    print("0. Importing LightZero to register policies...")
    import lzero
    print("✓ LightZero imported")
    
    # Test loading configuration
    print("1. Testing configuration import...")
    from zoo.game_2048.config.gat_stochastic_2048_config import game_2048_gat_stochastic_config as cfg
    print("✓ Configuration loaded successfully")
    
    print(f"  - Policy type: {cfg['policy']['type']}")
    print(f"  - Model observation shape: {cfg['policy']['model']['observation_shape']}")
    print(f"  - Action space size: {cfg['policy']['model']['action_space_size']}")
    
    # Test policy creation
    print("\n2. Testing policy creation...")
    from ding.utils import POLICY_REGISTRY
    
    PolicyClass = POLICY_REGISTRY.get(cfg['policy']['type'])
    if PolicyClass:
        print(f"✓ Policy class found: {PolicyClass}")
        
        # Create policy instance
        print("3. Testing policy instantiation...")
        try:
            policy = PolicyClass(cfg['policy'])
            print("✓ Policy instantiated successfully")
            print(f"  - Policy type: {type(policy)}")
        except Exception as e:
            print(f"✗ Error instantiating policy: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"✗ Policy class not found for type: {cfg['policy']['type']}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")
