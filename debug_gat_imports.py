#!/usr/bin/env python3
"""
Debug import script for GAT policies
"""

import sys
import os

# Add LightZero to Python path
sys.path.insert(0, '/opendilab/LightZero')

print("=== GAT Policy Import Debug ===")

# Test 1: Direct import of GAT stochastic policy
print("\n1. Testing direct import of GATStochasticMuZeroPolicy...")
try:
    from lzero.policy.gat_stochastic_muzero import GATStochasticMuZeroPolicy
    print("✓ Successfully imported GATStochasticMuZeroPolicy")
except Exception as e:
    print(f"✗ Error importing GATStochasticMuZeroPolicy: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Import from lzero.policy package
print("\n2. Testing import from lzero.policy package...")
try:
    from lzero.policy import GATStochasticMuZeroPolicy
    print("✓ Successfully imported GATStochasticMuZeroPolicy from lzero.policy")
except Exception as e:
    print(f"✗ Error importing from lzero.policy: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check policy registry
print("\n3. Testing policy registry...")
try:
    from ding.utils import POLICY_REGISTRY
    
    # List all registered policies
    print("Registered policies:")
    for key in POLICY_REGISTRY._registry.keys():
        print(f"  - {key}")
    
    # Check if our GAT policy is registered
    policy_class = POLICY_REGISTRY.get('gat_stochastic_muzero')
    if policy_class:
        print("✓ 'gat_stochastic_muzero' found in registry")
        print(f"  Policy class: {policy_class}")
    else:
        print("✗ 'gat_stochastic_muzero' not found in registry")
        
except Exception as e:
    print(f"✗ Error checking policy registry: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check model imports
print("\n4. Testing GAT model imports...")
try:
    from lzero.model.gat_stochastic_muzero_model import GATStochasticMuZeroModel
    print("✓ Successfully imported GATStochasticMuZeroModel")
except Exception as e:
    print(f"✗ Error importing GATStochasticMuZeroModel: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Debug Complete ===")
