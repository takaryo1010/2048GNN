#!/usr/bin/env python3
"""
Test policy registry for GAT policies
"""

import sys
sys.path.insert(0, '/opendilab/LightZero')

print("=== Policy Registry Test ===")

try:
    from ding.utils import POLICY_REGISTRY
    
    # Import the GAT policy to register it
    from lzero.policy.gat_stochastic_muzero import GATStochasticMuZeroPolicy
    
    print("Available methods on POLICY_REGISTRY:")
    print([method for method in dir(POLICY_REGISTRY) if not method.startswith('_')])
    
    # Try to get the policy
    try:
        policy_class = POLICY_REGISTRY.get('gat_stochastic_muzero')
        if policy_class:
            print("✓ 'gat_stochastic_muzero' found in registry")
            print(f"  Policy class: {policy_class}")
        else:
            print("✗ 'gat_stochastic_muzero' not found in registry")
    except Exception as e:
        print(f"Error getting policy: {e}")
    
    # Try to register and get again if needed
    print("\nTesting policy instantiation from registry...")
    try:
        # Get the policy class
        PolicyClass = POLICY_REGISTRY.get('gat_stochastic_muzero')
        if PolicyClass:
            print(f"✓ Policy class retrieved: {PolicyClass}")
            
            # Test if we can create a basic config
            print("Testing with basic config...")
            test_config = {
                'type': 'gat_stochastic_muzero',
                'model': {
                    'observation_shape': (16, 4, 4),
                    'action_space_size': 4,
                    'num_heads': 4,
                    'hidden_channels': 64,
                }
            }
            print("✓ Basic configuration created")
            
        else:
            print("✗ Could not retrieve policy class")
            
    except Exception as e:
        print(f"Error in policy instantiation test: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")
