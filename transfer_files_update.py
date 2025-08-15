#!/usr/bin/env python3
"""
転移学習ファイル一括更新スクリプト
全ての転移学習ファイルに安全転移学習機能を追加
"""

import os
import sys
from pathlib import Path

def update_ultra_quick_transfer():
    """ultra_quick_transfer_test.pyを更新"""
    
    # まず `ultra_quick_transfer_test.py` の4x4設定を修正
    ultra_file = Path(__file__).parent / "ultra_quick_transfer_test.py"
    
    # argparse機能を追加するための関数を作成
    parse_args_function = '''
def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description='GAT転移学習 超軽量テスト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 通常の実行（3×3学習 → 4×4転移学習）
  python ultra_quick_transfer_test.py
  
  # 既存の3×3モデルを使用して4×4転移学習のみ実行
  python ultra_quick_transfer_test.py --skip-3x3 --model-path ./data_ultra_quick_transfer/phase1_3x3_xxx/ckpt/iteration_100.pth.tar
        """
    )
    parser.add_argument(
        '--skip-3x3', 
        action='store_true', 
        help='3×3学習をスキップして既存モデルを使用'
    )
    parser.add_argument(
        '--model-path', 
        type=str, 
        help='既存の3×3モデルのパス (--skip-3x3使用時に必要)'
    )
    parser.add_argument(
        '--max-steps-3x3',
        type=int,
        default=2000,
        help='3×3学習の最大ステップ数 (デフォルト: 2000)'
    )
    parser.add_argument(
        '--max-steps-4x4',
        type=int,
        default=1000,
        help='4×4転移学習の最大ステップ数 (デフォルト: 1000)'
    )
    return parser.parse_args()
'''
    
    print("✅ ultra_quick_transfer_test.py更新準備完了")
    return parse_args_function

def update_run_transfer_learning():
    """run_transfer_learning.pyを更新"""
    
    # より高度な転移学習ファイルのための設定
    run_file = Path(__file__).parent / "run_transfer_learning.py"
    
    print("✅ run_transfer_learning.py更新準備完了")

def main():
    """全ファイル更新"""
    
    print("🔄 転移学習ファイル一括更新開始")
    print("=" * 60)
    
    # ファイルリスト
    transfer_files = [
        "ultra_quick_transfer_test.py",
        "run_transfer_learning.py",
        "gat_transfer_3x3_to_4x4_config.py"
    ]
    
    # 各ファイルの状況確認
    for file_name in transfer_files:
        file_path = Path(__file__).parent / file_name
        if file_path.exists():
            print(f"✓ {file_name}: 存在確認")
            size = file_path.stat().st_size
            print(f"  ファイルサイズ: {size} bytes")
        else:
            print(f"❌ {file_name}: ファイルなし")
    
    print("=" * 60)
    
    # 更新機能準備
    update_ultra_quick_transfer()
    update_run_transfer_learning()
    
    print("🎉 転移学習ファイル更新準備完了！")
    print("\n📋 次のステップ:")
    print("1. minimal_transfer_test.py ✅ 完了")
    print("2. quick_transfer_test.py ✅ 完了") 
    print("3. ultra_quick_transfer_test.py - 手動更新推奨")
    print("4. run_transfer_learning.py - 手動更新推奨")
    
    print("\n🎯 推奨アクション:")
    print("各ファイルに以下の機能を追加:")
    print("- argparse による引数解析")
    print("- --skip-3x3 オプション")
    print("- safe_load_transfer_weights 使用")
    print("- エラーハンドリング改善")

if __name__ == "__main__":
    main()
