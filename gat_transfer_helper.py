#!/usr/bin/env python3
"""
GAT転移学習ヘルパー - サイズ適応転移学習
"""

import torch
import torch.nn as nn
from collections import OrderedDict


def transfer_compatible_weights(source_dict, target_model, grid_size_change=False):
    """
    サイズが互換性のある重みのみを転移
    
    Args:
        source_dict: 転移元の state_dict
        target_model: 転移先のモデル
        grid_size_change: グリッドサイズが変更されたかどうか
    
    Returns:
        転移された重み辞書
    """
    target_dict = target_model.state_dict()
    transferred_dict = OrderedDict()
    
    print("\n🔄 GAT転移学習解析:")
    print("-" * 50)
    
    compatible_count = 0
    incompatible_count = 0
    
    for key, source_param in source_dict.items():
        if key in target_dict:
            target_param = target_dict[key]
            
            if source_param.shape == target_param.shape:
                transferred_dict[key] = source_param
                compatible_count += 1
                if 'gat' in key.lower():
                    print(f"✅ GAT層転移: {key} {source_param.shape}")
            else:
                print(f"❌ サイズ不一致: {key}")
                print(f"   転移元: {source_param.shape} → 転移先: {target_param.shape}")
                incompatible_count += 1
        else:
            print(f"⚠️  キー不存在: {key}")
            incompatible_count += 1
    
    print("-" * 50)
    print(f"📊 転移結果: 成功{compatible_count}個, 失敗{incompatible_count}個")
    
    # GAT関連の転移確認
    gat_transferred = sum(1 for k in transferred_dict.keys() if 'gat' in k.lower())
    print(f"🎯 GAT関連転移: {gat_transferred}個の層/パラメータ")
    
    return transferred_dict


def safe_load_transfer_weights(model, checkpoint_path):
    """
    安全な転移学習重み読み込み
    """
    print(f"\n🔄 安全転移学習開始: {checkpoint_path}")
    
    # チェックポイント読み込み
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in checkpoint:
        source_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        source_dict = checkpoint['state_dict']
    else:
        source_dict = checkpoint
    
    # 互換性のある重みのみ転移
    transferred_dict = transfer_compatible_weights(source_dict, model, grid_size_change=True)
    
    # 部分的な重み読み込み
    model_dict = model.state_dict()
    model_dict.update(transferred_dict)
    model.load_state_dict(model_dict)
    
    print("✅ 安全転移学習完了")
    return model


def analyze_gat_transfer(source_model, target_model):
    """
    GAT転移の詳細解析
    """
    print("\n🔍 GAT転移解析:")
    print("=" * 60)
    
    # GATConv層の確認
    def find_gat_layers(model, prefix=""):
        gat_info = []
        for name, module in model.named_modules():
            if 'GATConv' in str(type(module)):
                full_name = f"{prefix}.{name}" if prefix else name
                gat_info.append({
                    'name': full_name,
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'heads': module.heads
                })
        return gat_info
    
    source_gats = find_gat_layers(source_model, "source")
    target_gats = find_gat_layers(target_model, "target")
    
    print("📋 GAT層構造比較:")
    for i, (src, tgt) in enumerate(zip(source_gats, target_gats)):
        print(f"  GAT層{i+1}:")
        print(f"    転移元: {src['in_channels']}→{src['out_channels']} (ヘッド:{src['heads']})")
        print(f"    転移先: {tgt['in_channels']}→{tgt['out_channels']} (ヘッド:{tgt['heads']})")
        
        compatible = (src['in_channels'] == tgt['in_channels'] and 
                     src['out_channels'] == tgt['out_channels'] and 
                     src['heads'] == tgt['heads'])
        print(f"    転移可能: {'✅' if compatible else '❌'}")
    
    print("=" * 60)


if __name__ == "__main__":
    print("GAT転移学習ヘルパーモジュール")
