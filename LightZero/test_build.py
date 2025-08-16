#!/usr/bin/env python
"""
Test script to verify if LightZero can be imported and built correctly.
"""
import os
import sys

def check_files():
    """Check if required C++ files exist"""
    common_lib_path = "lzero/mcts/ctree/common_lib"
    required_files = ["cminimax.h", "cminimax.cpp", "utils.cpp"]
    
    print(f"Checking files in {common_lib_path}...")
    for file in required_files:
        filepath = os.path.join(common_lib_path, file)
        if os.path.exists(filepath):
            print(f"✓ {filepath} exists")
        else:
            print(f"✗ {filepath} missing")
            return False
    return True

def test_import():
    """Test if lzero can be imported"""
    try:
        import lzero
        print("✓ LightZero import successful")
        return True
    except ImportError as e:
        print(f"✗ LightZero import failed: {e}")
        return False

if __name__ == "__main__":
    print("=== LightZero Build Test ===")
    
    files_ok = check_files()
    if files_ok:
        import_ok = test_import()
        if import_ok:
            print("🎉 All tests passed!")
            sys.exit(0)
    
    print("❌ Some tests failed")
    sys.exit(1)
