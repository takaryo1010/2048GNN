
"""
CNN使用防止モジュール
GNNモデルで誤ってCNNを使用しようとした場合にエラーを発生させます
"""
import torch.nn as nn


class NoCNNAllowed(nn.Module):
    """
    CNNレイヤーの使用を防ぐプロキシクラス
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError(
            "❌ GNNモデルでCNNレイヤーの使用は禁止されています！\n"
            "このモデルはGraph Neural Network (GNN)ベースです。\n"
            "Conv2d, ResBlock, BatchNorm2dなどのCNNレイヤーは使用できません。\n"
            "chance_encoderのみ例外として許可されています。"
        )


# Conv2dの使用を防ぐ
class Conv2dNotAllowed(NoCNNAllowed):
    """Conv2dの使用を防止"""
    pass


# ResBlockの使用を防ぐ
class ResBlockNotAllowed(NoCNNAllowed):
    """ResBlockの使用を防止"""
    pass


# BatchNorm2dの使用を防ぐ  
class BatchNorm2dNotAllowed(NoCNNAllowed):
    """BatchNorm2dの使用を防止"""
    pass
