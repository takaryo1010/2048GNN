"""
より詳細なGNN使用検証スクリプト
中間出力を確認してGNNが本当に動いているか確認
"""
import torch
import sys
sys.path.append('LightZero')

from zoo.game_2048.config.stochastic_muzero_2048_gnn_config import main_config
from lzero.model.gnn_stochastic_muzero_model import GNNStochasticMuZeroModel

def test_representation_network():
    """
    RepresentationNetworkが本当にGNNを使っているか詳細テスト
    """
    print("\n" + "="*70)
    print("🔍 RepresentationNetworkの詳細テスト")
    print("="*70)
    
    # モデルをインスタンス化
    model = GNNStochasticMuZeroModel(**main_config.policy.model)
    model.eval()
    
    # RepresentationNetworkを取得
    repr_net = model.representation_network
    
    # GraphBuilderの存在確認
    print(f"\n✅ GraphBuilder存在: {hasattr(repr_net, 'graph_builder')}")
    print(f"✅ GraphSAGE存在: {hasattr(repr_net, 'gnn')}")
    
    if hasattr(repr_net, 'gnn'):
        gnn = repr_net.gnn
        print(f"✅ GraphSAGEの型: {type(gnn).__name__}")
        print(f"✅ GraphSAGEのレイヤー数: {len(gnn.convs) if hasattr(gnn, 'convs') else 'N/A'}")
    
    # ダミー入力
    batch_size = 2
    obs = torch.randn(batch_size, 16, 4, 4)
    
    print(f"\n入力形状: {obs.shape}")
    
    # ステップごとに追跡
    print("\n--- 順伝播の詳細 ---")
    
    with torch.no_grad():
        # 1. GraphBuilderでグラフに変換
        if hasattr(repr_net, 'graph_builder'):
            node_features, edge_index = repr_net.graph_builder.obs_to_graph(obs)
            print(f"1. GraphBuilder出力:")
            print(f"   - ノード特徴量形状: {node_features.shape}")
            print(f"   - エッジインデックス形状: {edge_index.shape}")
            print(f"   - エッジの最初の5個: {edge_index[:, :5].tolist()}")
        
        # 2. GNNを通す
        if hasattr(repr_net, 'gnn'):
            node_embeddings = repr_net.gnn(node_features, edge_index)
            print(f"2. GraphSAGE出力:")
            print(f"   - ノード埋め込み形状: {node_embeddings.shape}")
            print(f"   - 最初のノードの埋め込み統計:")
            print(f"     平均: {node_embeddings[0, 0].mean().item():.4f}")
            print(f"     標準偏差: {node_embeddings[0, 0].std().item():.4f}")
        
        # 3. 全体の出力
        latent_state = repr_net(obs)
        print(f"3. RepresentationNetwork出力:")
        print(f"   - 潜在状態形状: {latent_state.shape}")
        print(f"   - 潜在状態統計:")
        print(f"     平均: {latent_state.mean().item():.4f}")
        print(f"     標準偏差: {latent_state.std().item():.4f}")
    
    return True


def test_graphsage_execution():
    """
    GraphSAGEの各レイヤーが実行されているか確認
    """
    print("\n" + "="*70)
    print("🔍 GraphSAGEレイヤーの実行確認")
    print("="*70)
    
    model = GNNStochasticMuZeroModel(**main_config.policy.model)
    model.eval()
    
    repr_net = model.representation_network
    
    if not hasattr(repr_net, 'gnn'):
        print("❌ GNNが見つかりません")
        return False
    
    gnn = repr_net.gnn
    
    # 各レイヤーにフックを設定
    layer_outputs = {}
    
    def make_hook(layer_name):
        def hook(module, input, output):
            layer_outputs[layer_name] = {
                'input_shape': input[0].shape if isinstance(input, tuple) else 'N/A',
                'output_shape': output.shape if hasattr(output, 'shape') else 'N/A',
                'executed': True
            }
        return hook
    
    # GraphSAGEConvレイヤーにフックを登録
    hooks = []
    if hasattr(gnn, 'convs'):
        for i, conv in enumerate(gnn.convs):
            hook = conv.register_forward_hook(make_hook(f'GraphSAGEConv_{i}'))
            hooks.append(hook)
    
    # ダミー入力で実行
    obs = torch.randn(2, 16, 4, 4)
    
    with torch.no_grad():
        output = repr_net(obs)
    
    # フックを削除
    for hook in hooks:
        hook.remove()
    
    # 結果を表示
    print(f"\n実行されたGraphSAGEConvレイヤー: {len(layer_outputs)}")
    for layer_name, info in layer_outputs.items():
        print(f"\n{layer_name}:")
        print(f"  入力形状: {info['input_shape']}")
        print(f"  出力形状: {info['output_shape']}")
        print(f"  実行済み: {info['executed']}")
    
    return len(layer_outputs) > 0


def compare_gnn_vs_random():
    """
    GNNの出力が単なるランダムではないことを確認
    同じ入力に対して同じ出力が得られるか
    """
    print("\n" + "="*70)
    print("🔍 GNN出力の一貫性テスト")
    print("="*70)
    
    model = GNNStochasticMuZeroModel(**main_config.policy.model)
    model.eval()
    
    # 同じ入力を2回通す
    obs = torch.randn(1, 16, 4, 4)
    
    with torch.no_grad():
        output1 = model.representation_network(obs)
        output2 = model.representation_network(obs)
    
    # 差を計算
    diff = (output1 - output2).abs().max().item()
    
    print(f"\n同じ入力に対する出力の差: {diff}")
    print(f"一貫性テスト: {'✅ 合格' if diff < 1e-6 else '❌ 不合格'}")
    
    # 異なる入力で異なる出力が得られるか
    obs2 = torch.randn(1, 16, 4, 4)
    
    with torch.no_grad():
        output3 = model.representation_network(obs2)
    
    diff2 = (output1 - output3).abs().mean().item()
    
    print(f"\n異なる入力に対する出力の差: {diff2}")
    print(f"変化テスト: {'✅ 合格' if diff2 > 0.01 else '❌ 不合格'}")
    
    return diff < 1e-6 and diff2 > 0.01


def test_edge_influence():
    """
    エッジ接続がGNNの出力に影響を与えているか確認
    """
    print("\n" + "="*70)
    print("🔍 エッジ接続の影響テスト")
    print("="*70)
    
    model = GNNStochasticMuZeroModel(**main_config.policy.model)
    repr_net = model.representation_network
    
    # 元のエッジを保存
    original_edges = repr_net.graph_builder.edge_index.clone()
    
    print(f"元のエッジ数: {original_edges.shape[1]}")
    
    # 通常の出力
    obs = torch.randn(1, 16, 4, 4)
    with torch.no_grad():
        output_normal = repr_net(obs)
    
    # エッジを減らして実行（隣接ノードのみ）
    # 4x4グリッドで隣接エッジのみ（上下左右）
    adjacent_edges = []
    for i in range(4):
        for j in range(4):
            node_id = i * 4 + j
            # 右
            if j < 3:
                neighbor = i * 4 + (j + 1)
                adjacent_edges.append([node_id, neighbor])
                adjacent_edges.append([neighbor, node_id])
            # 下
            if i < 3:
                neighbor = (i + 1) * 4 + j
                adjacent_edges.append([node_id, neighbor])
                adjacent_edges.append([neighbor, node_id])
    
    adjacent_edge_index = torch.tensor(adjacent_edges, dtype=torch.long).t().contiguous()
    
    print(f"隣接エッジのみの数: {adjacent_edge_index.shape[1]}")
    
    # エッジを置き換え
    repr_net.graph_builder.edge_index = adjacent_edge_index
    
    with torch.no_grad():
        output_adjacent = repr_net(obs)
    
    # エッジを復元
    repr_net.graph_builder.edge_index = original_edges
    
    # 差を計算
    diff = (output_normal - output_adjacent).abs().mean().item()
    
    print(f"\nエッジ構造を変えた時の出力の差: {diff}")
    print(f"エッジ影響テスト: {'✅ 合格（エッジが影響している）' if diff > 0.01 else '⚠️  エッジの影響が小さい'}")
    
    return diff > 0.01


def main():
    print("\n" + "="*70)
    print("詳細GNN検証スクリプト")
    print("="*70)
    
    # テストを実行
    tests = [
        ("RepresentationNetworkの詳細確認", test_representation_network),
        ("GraphSAGEレイヤーの実行確認", test_graphsage_execution),
        ("GNN出力の一貫性", compare_gnn_vs_random),
        ("エッジ接続の影響", test_edge_influence),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} でエラー: {e}")
            results.append((test_name, False))
    
    # 最終結果
    print("\n" + "="*70)
    print("📊 最終結果")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    
    print()
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print(f"\n合計: {passed}/{len(results)} のテストに合格")
    
    if passed == len(results):
        print("\n🎉 結論: このモデルは確実にGNNを使用し、正常に動作しています！")
    elif passed >= len(results) - 1:
        print("\n✅ 結論: このモデルはGNNを使用しています")
    else:
        print("\n⚠️  結論: GNNの使用に問題がある可能性があります")


if __name__ == "__main__":
    main()
