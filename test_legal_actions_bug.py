"""
get_legal_actions のバグを検証するスクリプト

バグ：get_legal_actions() で _move() を呼び出すと、
_merge_line() 内で self.score が変更されるが、復元されない。
"""

import sys
import numpy as np

sys.path.append('./LightZero')

# CNNエミュレーターから環境をインポート
from cnn_any_size_emulator import Game2048

def test_get_legal_actions_bug():
    """get_legal_actions のバグを検証"""
    print("="*70)
    print("get_legal_actions バグ検証")
    print("="*70)
    
    env = Game2048()
    
    # 特定の盤面を設定
    env.board = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 0]
    ], dtype=np.int32)
    env.score = 1000
    
    print("\n初期状態:")
    print(f"Board:\n{env.board}")
    print(f"Score: {env.score}")
    
    # 合法アクションを取得
    print("\nget_legal_actions() を呼び出し...")
    legal_actions = env.get_legal_actions()
    
    print(f"\n合法アクション: {legal_actions}")
    print(f"Board:\n{env.board}")
    print(f"Score: {env.score}")
    
    if env.score != 1000:
        print("\n❌ バグ検出！スコアが変更されました！")
        print(f"   期待値: 1000")
        print(f"   実際の値: {env.score}")
        return False
    else:
        print("\n✓ OK: スコアは変更されていません")
        return True


def test_multiple_calls():
    """複数回呼び出しをテスト"""
    print("\n" + "="*70)
    print("複数回呼び出しテスト")
    print("="*70)
    
    env = Game2048()
    env.board = np.array([
        [2, 2, 0, 0],
        [4, 4, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.int32)
    env.score = 0
    
    print(f"\n初期スコア: {env.score}")
    
    for i in range(5):
        legal_actions = env.get_legal_actions()
        print(f"呼び出し {i+1}: スコア = {env.score}, 合法アクション = {legal_actions}")
    
    if env.score != 0:
        print(f"\n❌ バグ検出！スコアが {env.score} に変更されました！")
        return False
    else:
        print("\n✓ OK: スコアは0のままです")
        return True


def test_with_gameplay():
    """実際のゲームプレイでテスト"""
    print("\n" + "="*70)
    print("ゲームプレイテスト")
    print("="*70)
    
    env = Game2048()
    env.reset()
    
    scores = []
    max_tiles = []
    
    for episode in range(10):
        env.reset()
        done = False
        step = 0
        
        while not done and step < 1000:
            legal_actions = env.get_legal_actions()
            
            if len(legal_actions) == 0:
                break
            
            # ランダムにアクションを選択
            action = legal_actions[np.random.randint(len(legal_actions))]
            obs, reward, done, info = env.step(action)
            step += 1
        
        scores.append(env.score)
        max_tiles.append(env.max_tile)
    
    print(f"\n10エピソードの結果:")
    print(f"平均スコア: {np.mean(scores):.1f}")
    print(f"平均最大タイル: {np.mean(max_tiles):.1f}")
    print(f"最大タイル分布:")
    
    tile_counts = {}
    for tile in max_tiles:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    
    for tile in sorted(tile_counts.keys(), reverse=True):
        print(f"  {tile}: {tile_counts[tile]}回")
    
    # 8以下で終わったエピソードがあるかチェック
    low_tiles = [t for t in max_tiles if t <= 8]
    if low_tiles:
        print(f"\n⚠️ 警告: {len(low_tiles)}エピソードが8以下のタイルで終了しました")
        print(f"   これは異常です（通常は少なくとも16以上になるはず）")
        return False
    else:
        print("\n✓ OK: すべてのエピソードで16以上のタイルを達成")
        return True


def main():
    print("CNNエミュレーター get_legal_actions バグ検証\n")
    
    test1 = test_get_legal_actions_bug()
    test2 = test_multiple_calls()
    test3 = test_with_gameplay()
    
    print("\n" + "="*70)
    print("検証結果サマリー")
    print("="*70)
    print(f"基本テスト: {'✓ PASS' if test1 else '❌ FAIL'}")
    print(f"複数回呼び出しテスト: {'✓ PASS' if test2 else '❌ FAIL'}")
    print(f"ゲームプレイテスト: {'✓ PASS' if test3 else '❌ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n✓ すべてのテストに合格しました！バグは修正されています。")
    else:
        print("\n❌ 一部のテストが失敗しました。バグが残っている可能性があります。")


if __name__ == '__main__':
    main()
