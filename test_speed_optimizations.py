#!/usr/bin/env python3
"""
GAT速度最適化テスト
超簡単セット（D-1, D-2, D-3）の効果を測定

最適化内容:
- D-1: インプレース演算 (inplace=True)
- D-2: Mixed Precision (FP16)
- D-3: torch.compile()

期待効果: 30-40%の高速化
"""
import sys
import time
import torch
import torch.nn as nn
sys.path.insert(0, 'LightZero')

from lzero.model.gat_stochastic_muzero_model import (
    GATStochasticMuZeroModel,
    optimize_gat_model_for_speed
)


def benchmark_model(model, obs, num_iterations=100, warmup=10, use_amp=False):
    """
    モデルの推論速度をベンチマーク
    
    Args:
        model: テストするモデル
        obs: 入力観測
        num_iterations: 測定イテレーション数
        warmup: ウォームアップイテレーション数
        use_amp: Mixed Precision (AMP)を使用するか
    
    Returns:
        steps_per_sec: 1秒あたりの処理ステップ数
    """
    model.eval()
    device = obs.device
    
    # ウォームアップ
    print(f"  ウォームアップ中 ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model.initial_inference(obs)
            else:
                _ = model.initial_inference(obs)
    
    # 測定
    print(f"  ベンチマーク実行中 ({num_iterations} iterations)...")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model.initial_inference(obs)
            else:
                _ = model.initial_inference(obs)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    steps_per_sec = num_iterations / elapsed_time
    
    return steps_per_sec


def main():
    print("=" * 80)
    print("GAT速度最適化テスト - 超簡単セット")
    print("=" * 80)
    print()
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📍 デバイス: {device}")
    
    if device.type == 'cpu':
        print("⚠️  CPUモードです。GPUを使用すると最適化効果がより顕著になります")
    
    print()
    
    # バッチサイズ
    batch_size = 256
    obs = torch.randn(batch_size, 16, 4, 4, device=device)
    
    print(f"📊 テスト設定:")
    print(f"  バッチサイズ: {batch_size}")
    print(f"  入力形状: {obs.shape}")
    print()
    
    # ========================================================================
    # テスト1: ベースライン（最適化なし）
    # ========================================================================
    print("🔵 テスト1: ベースライン（最適化前のコード）")
    print("-" * 80)
    
    model_baseline = GATStochasticMuZeroModel(
        observation_shape=(16, 4, 4),
        action_space_size=4,
        chance_space_size=16,
        num_channels=128,
        num_gnn_layers=3,
        num_heads=4,
        edge_mode='adjacent',  # 既存のB-1最適化
        norm_type='layer',      # 既存のB-3最適化
    ).to(device)
    
    speed_baseline = benchmark_model(model_baseline, obs)
    print(f"✅ ベースライン速度: {speed_baseline:.2f} steps/sec")
    print()
    
    # ========================================================================
    # テスト2: D-1のみ（インプレース演算）
    # ========================================================================
    print("🟢 テスト2: D-1最適化（インプレース演算）")
    print("-" * 80)
    print("  ※ 既に実装済み（コード内のinplace=True）")
    
    # D-1は既にコードに組み込まれているので、同じモデルを使用
    speed_d1 = speed_baseline  # 既に適用済み
    speedup_d1 = (speed_d1 / speed_baseline - 1) * 100
    print(f"✅ D-1適用後: {speed_d1:.2f} steps/sec (ベースライン比: {speedup_d1:+.1f}%)")
    print()
    
    # ========================================================================
    # テスト3: D-2（Mixed Precision）
    # ========================================================================
    print("🟡 テスト3: D-2最適化（Mixed Precision FP16）")
    print("-" * 80)
    
    if device.type == 'cuda':
        speed_d2 = benchmark_model(model_baseline, obs, use_amp=True)
        speedup_d2 = (speed_d2 / speed_baseline - 1) * 100
        print(f"✅ D-2適用後: {speed_d2:.2f} steps/sec (ベースライン比: {speedup_d2:+.1f}%)")
    else:
        print("⚠️  Mixed PrecisionはCUDA必須のためスキップ")
        speed_d2 = speed_baseline
        speedup_d2 = 0.0
    
    print()
    
    # ========================================================================
    # テスト4: D-3（torch.compile）
    # ========================================================================
    print("🟣 テスト4: D-3最適化（torch.compile）")
    print("-" * 80)
    
    try:
        if hasattr(torch, 'compile'):
            model_compiled = GATStochasticMuZeroModel(
                observation_shape=(16, 4, 4),
                action_space_size=4,
                chance_space_size=16,
                num_channels=128,
                num_gnn_layers=3,
                num_heads=4,
                edge_mode='adjacent',
                norm_type='layer',
            ).to(device)
            
            # ヘルパー関数を使用
            model_compiled = optimize_gat_model_for_speed(
                model_compiled,
                use_mixed_precision=False,  # まずcompileのみテスト
                use_compile=True,
                compile_mode='default'
            )
            
            print()
            speed_d3 = benchmark_model(model_compiled, obs)
            speedup_d3 = (speed_d3 / speed_baseline - 1) * 100
            print(f"✅ D-3適用後: {speed_d3:.2f} steps/sec (ベースライン比: {speedup_d3:+.1f}%)")
        else:
            print("⚠️  PyTorch 2.0+が必要です。torch.compile()をスキップ")
            speed_d3 = speed_baseline
            speedup_d3 = 0.0
    except Exception as e:
        print(f"⚠️  torch.compile()エラー: {e}")
        speed_d3 = speed_baseline
        speedup_d3 = 0.0
    
    print()
    
    # ========================================================================
    # テスト5: すべての最適化を組み合わせ
    # ========================================================================
    print("🔴 テスト5: フル最適化（D-1 + D-2 + D-3）")
    print("-" * 80)
    
    try:
        if device.type == 'cuda' and hasattr(torch, 'compile'):
            model_full = GATStochasticMuZeroModel(
                observation_shape=(16, 4, 4),
                action_space_size=4,
                chance_space_size=16,
                num_channels=128,
                num_gnn_layers=3,
                num_heads=4,
                edge_mode='adjacent',
                norm_type='layer',
            ).to(device)
            
            # すべての最適化を適用
            model_full = optimize_gat_model_for_speed(
                model_full,
                use_mixed_precision=True,
                use_compile=True,
                compile_mode='default'
            )
            
            print()
            speed_full = benchmark_model(model_full, obs, use_amp=True)
            speedup_full = (speed_full / speed_baseline - 1) * 100
            print(f"✅ フル最適化: {speed_full:.2f} steps/sec (ベースライン比: {speedup_full:+.1f}%)")
        else:
            print("⚠️  CUDAとPyTorch 2.0+が必要です")
            speed_full = max(speed_d2, speed_d3)
            speedup_full = (speed_full / speed_baseline - 1) * 100
    except Exception as e:
        print(f"⚠️  フル最適化エラー: {e}")
        speed_full = max(speed_d2, speed_d3)
        speedup_full = (speed_full / speed_baseline - 1) * 100
    
    print()
    
    # ========================================================================
    # 結果サマリー
    # ========================================================================
    print("=" * 80)
    print("📊 結果サマリー")
    print("=" * 80)
    print()
    print(f"{'最適化':<30} {'速度 (steps/sec)':<20} {'高速化率':<15}")
    print("-" * 80)
    print(f"{'ベースライン（最適化なし）':<30} {speed_baseline:>10.2f}          {0.0:>10.1f}%")
    print(f"{'D-1: インプレース演算':<30} {speed_d1:>10.2f}          {speedup_d1:>10.1f}%")
    print(f"{'D-2: Mixed Precision (FP16)':<30} {speed_d2:>10.2f}          {speedup_d2:>10.1f}%")
    print(f"{'D-3: torch.compile()':<30} {speed_d3:>10.2f}          {speedup_d3:>10.1f}%")
    print(f"{'フル最適化 (D-1+D-2+D-3)':<30} {speed_full:>10.2f}          {speedup_full:>10.1f}%")
    print()
    
    # 期待値との比較
    expected_speedup = 30.0  # 30-40%の中央値
    print(f"📈 期待される高速化率: ~{expected_speedup:.0f}%")
    print(f"📈 実際の高速化率: {speedup_full:.1f}%")
    
    if speedup_full >= expected_speedup:
        print("✅ 期待通りまたはそれ以上の高速化が達成されました！")
    elif speedup_full >= expected_speedup * 0.7:
        print("⚠️  期待値の70%以上の高速化が達成されました")
    else:
        print("❌ 期待値を下回っています。環境や設定を確認してください")
    
    print()
    print("=" * 80)
    print("💡 使用方法:")
    print("=" * 80)
    print("""
トレーニングスクリプトで以下のように使用:

from lzero.model.gat_stochastic_muzero_model import (
    GATStochasticMuZeroModel,
    optimize_gat_model_for_speed
)

# モデル作成
model = GATStochasticMuZeroModel(...)

# 最適化適用
model = optimize_gat_model_for_speed(
    model,
    use_mixed_precision=True,
    use_compile=True,
    compile_mode='default'  # または 'max-autotune' で最大最適化
)

# Mixed Precision用のスケーラー
scaler = torch.cuda.amp.GradScaler()

# トレーニングループ
for batch in dataloader:
    with torch.cuda.amp.autocast():
        output = model(batch)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
""")


if __name__ == '__main__':
    main()
