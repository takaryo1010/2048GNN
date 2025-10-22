"""
GNN使用の検証スクリプト
このスクリプトは、モデルが本当にGNNを使用しているか確認します。
"""
import torch
import sys
import os
sys.path.append('LightZero')

# 設定ファイルをインポート
from zoo.game_2048.config.stochastic_muzero_2048_gnn_config import main_config, create_config

# モデルをインポート
from lzero.model.gnn_stochastic_muzero_model import GNNStochasticMuZeroModel

def check_gnn_layers(model):
    """
    モデル内のGNN関連レイヤーを検出
    """
    print("\n" + "="*70)
    print("🔍 モデル内のGNN/CNN レイヤーの検証")
    print("="*70)
    
    gnn_layers = []
    cnn_layers = []
    
    # すべてのモジュールを調べる
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        # GNN関連のレイヤーを検出
        if 'GraphSAGE' in module_type or 'GraphBuilder' in module_type or 'GNN' in module_type:
            gnn_layers.append((name, module_type))
        
        # CNN関連のレイヤーを検出
        if 'Conv2d' in module_type or 'ResBlock' in module_type:
            cnn_layers.append((name, module_type))
    
    # 結果表示
    print(f"\n✅ GNN関連レイヤー数: {len(gnn_layers)}")
    for name, mtype in gnn_layers[:10]:  # 最初の10個を表示
        print(f"   - {name}: {mtype}")
    if len(gnn_layers) > 10:
        print(f"   ... ({len(gnn_layers) - 10} more)")
    
    print(f"\n❌ CNN関連レイヤー数: {len(cnn_layers)}")
    if cnn_layers:
        for name, mtype in cnn_layers:
            print(f"   - {name}: {mtype}")
    else:
        print("   (CNNレイヤーは見つかりませんでした)")
    
    return len(gnn_layers) > 0, len(cnn_layers) == 0


def trace_forward_pass(model):
    """
    順伝播時にどのレイヤーが使われているか追跡
    """
    print("\n" + "="*70)
    print("🔍 順伝播パスの追跡")
    print("="*70)
    
    # フックでレイヤーの実行を追跡
    executed_layers = []
    
    def hook_fn(module, input, output):
        layer_type = type(module).__name__
        executed_layers.append(layer_type)
    
    # すべてのモジュールにフックを登録
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 葉ノードのみ
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    # ダミー入力で順伝播
    batch_size = 2
    obs = torch.randn(batch_size, 16, 4, 4)  # [B, C, H, W]
    
    with torch.no_grad():
        model.eval()
        # initial_inferenceを呼び出す
        output = model.initial_inference(obs)
    
    # フックを削除
    for hook in hooks:
        hook.remove()
    
    # 実行されたレイヤーを集計
    layer_counts = {}
    for layer in executed_layers:
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    print("\n実行されたレイヤーの種類:")
    for layer_type, count in sorted(layer_counts.items(), key=lambda x: -x[1]):
        print(f"   {layer_type}: {count}回")
    
    # GNN/CNNの確認
    gnn_executed = any('GraphSAGE' in k or 'GNN' in k for k in layer_counts.keys())
    cnn_executed = any('Conv2d' in k or 'ResBlock' in k for k in layer_counts.keys())
    
    print(f"\n✅ GraphSAGE/GNN レイヤーが実行された: {gnn_executed}")
    print(f"❌ Conv2d/CNN レイヤーが実行された: {cnn_executed}")
    
    return gnn_executed, not cnn_executed


def check_edge_construction(model):
    """
    グラフのエッジ構築を確認
    """
    print("\n" + "="*70)
    print("🔍 グラフ構造の確認")
    print("="*70)
    
    # RepresentationNetworkにアクセス
    repr_net = model.representation_network
    
    # GraphBuilderを取得
    if hasattr(repr_net, 'graph_builder'):
        graph_builder = repr_net.graph_builder
        edge_index = graph_builder.edge_index
        
        print(f"\nグリッドサイズ: {graph_builder.grid_size}x{graph_builder.grid_size}")
        print(f"ノード数: {graph_builder.num_nodes}")
        print(f"エッジモード: {graph_builder.edge_mode}")
        print(f"エッジ数: {edge_index.shape[1]}")
        print(f"エッジインデックス形状: {edge_index.shape}")
        
        # エッジの一部を表示
        print(f"\n最初の10個のエッジ:")
        for i in range(min(10, edge_index.shape[1])):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            print(f"   エッジ {i}: ノード {src} -> ノード {dst}")
        
        return True
    else:
        print("❌ GraphBuilderが見つかりません")
        return False


def compare_with_cnn_config():
    """
    CNN版の設定と比較
    """
    print("\n" + "="*70)
    print("🔍 設定ファイルの確認")
    print("="*70)
    
    print(f"\nmodel_type: {main_config.policy.model.get('model_type', 'Not specified')}")
    print(f"GNNレイヤー数: {main_config.policy.model.get('num_gnn_layers', 'Not specified')}")
    print(f"GNN隠れ次元: {main_config.policy.model.get('num_channels', 'Not specified')}")
    print(f"エッジモード: {main_config.policy.model.get('edge_mode', 'Not specified')}")
    
    # モデルタイプの確認
    model_type = create_config.model.type
    print(f"\nモデル登録タイプ: {model_type}")
    
    is_gnn = 'GNN' in model_type or 'gnn' in main_config.policy.model.get('model_type', '')
    
    return is_gnn


def main():
    print("\n" + "="*70)
    print("GNN使用検証スクリプト")
    print("="*70)
    
    # 1. 設定ファイルの確認
    config_is_gnn = compare_with_cnn_config()
    
    # 2. モデルをインスタンス化
    print("\n" + "="*70)
    print("🔧 モデルのインスタンス化")
    print("="*70)
    
    try:
        model = GNNStochasticMuZeroModel(**main_config.policy.model)
        print("✅ GNNStochasticMuZeroModel のインスタンス化に成功")
    except Exception as e:
        print(f"❌ モデルのインスタンス化に失敗: {e}")
        return
    
    # 3. GNN/CNNレイヤーの検出
    has_gnn, no_cnn = check_gnn_layers(model)
    
    # 4. グラフ構造の確認
    has_graph = check_edge_construction(model)
    
    # 5. 順伝播の追跡
    gnn_exec, no_cnn_exec = trace_forward_pass(model)
    
    # 最終判定
    print("\n" + "="*70)
    print("📊 最終判定")
    print("="*70)
    
    checks = [
        ("設定ファイルでGNNが指定されている", config_is_gnn),
        ("モデルにGNNレイヤーが存在する", has_gnn),
        ("モデルにCNNレイヤーが存在しない", no_cnn),
        ("グラフ構造（エッジ）が構築されている", has_graph),
        ("順伝播時にGNNレイヤーが実行される", gnn_exec),
        ("順伝播時にCNNレイヤーが実行されない", no_cnn_exec),
    ]
    
    print()
    passed = 0
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    print(f"\n合計: {passed}/{len(checks)} のチェックに合格")
    
    if passed == len(checks):
        print("\n🎉 結論: このモデルは確実にGNNを使用しています！")
    elif passed >= len(checks) - 1:
        print("\n⚠️  結論: このモデルはGNNを使用している可能性が高いです")
    else:
        print("\n❌ 結論: このモデルがGNNを使用しているか不明確です")
    
    # パラメータ数の比較
    print("\n" + "="*70)
    print("📈 パラメータ数")
    print("="*70)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n総パラメータ数: {total_params:,}")
    print(f"訓練可能パラメータ数: {trainable_params:,}")


if __name__ == "__main__":
    main()
