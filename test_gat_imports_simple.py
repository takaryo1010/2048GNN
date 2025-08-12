#!/usr/bin/env python3
"""
Simple test to check if GAT imports work without circular import issues
"""

import sys
import os

# Add LightZero to Python path
sys.path.insert(0, '/opendilab/LightZero')

print("=== GAT インポート問題解決テスト ===")

# Test 1: Direct model import
print("\n1. GATモデル直接インポートテスト...")
try:
    from lzero.model.gat_stochastic_muzero_model import GATStochasticMuZeroModel
    print("✓ GATStochasticMuZeroModel直接インポート成功")
except Exception as e:
    print(f"✗ GATモデルインポートエラー: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Policy import
print("\n2. GATポリシー直接インポートテスト...")
try:
    from lzero.policy.gat_stochastic_muzero import GATStochasticMuZeroPolicy
    print("✓ GATStochasticMuZeroPolicy直接インポート成功")
except Exception as e:
    print(f"✗ GATポリシーインポートエラー: {e}")
    print("これは循環インポートの問題である可能性があります")
    import traceback
    traceback.print_exc()

# Test 3: Check if we can import MCTS classes separately
print("\n3. MCTS個別インポートテスト...")
try:
    from lzero.mcts.tree_search.mcts_ctree_stochastic import StochasticMuZeroMCTSCtree
    print("✓ StochasticMuZeroMCTSCtree個別インポート成功")
except Exception as e:
    print(f"✗ MCTSインポートエラー: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Import order test  
print("\n4. インポート順序テスト...")
try:
    # First import MCTS
    from lzero.mcts.tree_search.mcts_ctree_stochastic import StochasticMuZeroMCTSCtree
    from lzero.mcts.tree_search.mcts_ptree_stochastic import StochasticMuZeroMCTSPtree
    print("✓ MCTS先行インポート成功")
    
    # Then import policy
    from lzero.policy.gat_stochastic_muzero import GATStochasticMuZeroPolicy
    print("✓ MCTS先行後のGATポリシーインポート成功")
    
except Exception as e:
    print(f"✗ インポート順序テストエラー: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Configuration import
print("\n5. 設定ファイルインポートテスト...")
try:
    from zoo.game_2048.config.gat_stochastic_2048_config import main_config, create_config
    print("✓ GAT設定ファイルインポート成功")
    print(f"  - ポリシータイプ: {main_config.policy.type}")
    print(f"  - モデルタイプ: {main_config.policy.model.model}")
except Exception as e:
    print(f"✗ 設定ファイルインポートエラー: {e}")
    import traceback
    traceback.print_exc()

print("\n=== テスト完了 ===")
print("\n推奨解決策:")
print("1. 循環インポートが発生する場合、GATポリシーを直接インポートする")
print("2. 設定でmodel='gat_stochastic'を指定して、モデルレベルでGATを有効化する")
print("3. standard stochastic_muzero policyを使用し、model設定でGATを指定する")
