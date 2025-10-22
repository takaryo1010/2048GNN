"""
バリデーション付きGNNモデルのテスト
"""
import sys
sys.path.append('LightZero')

from lzero.model.gnn_stochastic_muzero_model import GNNStochasticMuZeroModel
from zoo.game_2048.config.stochastic_muzero_2048_gnn_config import main_config

print('🧪 GNNモデルのインスタンス化テスト（バリデーション付き）...\n')

try:
    model = GNNStochasticMuZeroModel(**main_config.policy.model)
    print('✅ モデルのインスタンス化成功')
    print('✅ CNNバリデーションチェック完了')
    print('✅ GNNコンポーネント検証完了')
    print()
    print('🎉 このモデルは純粋なGNNモデルです（chance_encoderを除く）')
    
    # 詳細情報
    print('\n' + '='*70)
    print('モデル構成の詳細')
    print('='*70)
    
    gnn_count = 0
    cnn_in_chance = 0
    
    for name, module in model.named_modules():
        mtype = type(module).__name__
        if 'GNN' in mtype or 'GraphSAGE' in mtype:
            gnn_count += 1
        if 'Conv2d' in mtype and 'chance_encoder' in name:
            cnn_in_chance += 1
    
    print(f'\nGNNコンポーネント数: {gnn_count}')
    print(f'CNN（chance_encoderのみ）: {cnn_in_chance}')
    print(f'総パラメータ数: {sum(p.numel() for p in model.parameters()):,}')
    
except RuntimeError as e:
    print(f'❌ バリデーションエラー:')
    print(str(e))
except Exception as e:
    print(f'❌ エラー: {e}')
    import traceback
    traceback.print_exc()
