"""
GAT StochasticMuZero 転移学習ヘルパー
3×3から4×4への転移学習をサポート
"""

import torch
import os
import glob
from typing import Optional, Tuple
import logging


class TransferLearningHelper:
    """GAT StochasticMuZeroの転移学習をサポートするヘルパークラス"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def find_latest_checkpoint(self, experiment_dir: str) -> Optional[str]:
        """
        最新のチェックポイントファイルを検索
        
        Args:
            experiment_dir: 実験ディレクトリのパス
            
        Returns:
            最新のチェックポイントファイルのパス、見つからない場合はNone
        """
        search_pattern = os.path.join(experiment_dir, "**/ckpt_*.pth.tar")
        checkpoint_files = glob.glob(search_pattern, recursive=True)
        
        if not checkpoint_files:
            self.logger.warning(f"チェックポイントが見つかりません: {experiment_dir}")
            return None
            
        # 最新のファイルを取得
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        self.logger.info(f"最新のチェックポイント発見: {latest_checkpoint}")
        return latest_checkpoint
    
    def extract_grid_size_from_config(self, config) -> int:
        """設定からグリッドサイズを抽出"""
        obs_shape = config.policy.model.observation_shape
        if len(obs_shape) == 3:
            # (channels, height, width) 形式
            return obs_shape[1]  # height = width = grid_size
        else:
            raise ValueError(f"想定外の観測形状: {obs_shape}")
    
    def load_checkpoint_for_transfer(self, checkpoint_path: str, 
                                   target_model, source_grid_size: int, 
                                   target_grid_size: int) -> dict:
        """
        転移学習用にチェックポイントを読み込み
        
        Args:
            checkpoint_path: チェックポイントファイルのパス
            target_model: ターゲットモデル
            source_grid_size: ソースのグリッドサイズ
            target_grid_size: ターゲットのグリッドサイズ
            
        Returns:
            転移学習情報を含む辞書
        """
        self.logger.info(f"転移学習用チェックポイント読み込み: {checkpoint_path}")
        
        try:
            # チェックポイントを読み込み
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # モデルの状態辞書を取得
            if 'model' in checkpoint:
                model_state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                model_state_dict = checkpoint['state_dict']
            else:
                # チェックポイント全体がモデルの状態辞書の可能性
                model_state_dict = checkpoint
            
            # 転移学習用にモデルの状態辞書を読み込み
            if hasattr(target_model, 'load_state_dict_for_transfer'):
                target_model.load_state_dict_for_transfer(
                    model_state_dict, source_grid_size, target_grid_size
                )
            else:
                # フォールバック: 通常の読み込み（サイズ不一致は無視）
                self.logger.warning("カスタム転移学習メソッドが見つかりません。通常読み込みにフォールバック")
                target_model.load_state_dict(model_state_dict, strict=False)
            
            transfer_info = {
                'source_checkpoint': checkpoint_path,
                'source_grid_size': source_grid_size,
                'target_grid_size': target_grid_size,
                'transfer_success': True,
                'original_checkpoint_keys': list(checkpoint.keys())
            }
            
            if 'train_iter' in checkpoint:
                transfer_info['source_train_iter'] = checkpoint['train_iter']
            if 'envstep' in checkpoint:
                transfer_info['source_envstep'] = checkpoint['envstep']
                
            self.logger.info("✅ 転移学習用モデル読み込み完了")
            return transfer_info
            
        except Exception as e:
            self.logger.error(f"❌ 転移学習用チェックポイント読み込みエラー: {e}")
            return {
                'source_checkpoint': checkpoint_path,
                'source_grid_size': source_grid_size,
                'target_grid_size': target_grid_size,
                'transfer_success': False,
                'error': str(e)
            }
    
    def validate_transfer_compatibility(self, source_config, target_config) -> Tuple[bool, str]:
        """
        転移学習の互換性を検証
        
        Args:
            source_config: ソース設定
            target_config: ターゲット設定
            
        Returns:
            (互換性があるか, メッセージ)
        """
        try:
            # GAT設定の確認
            source_gat = source_config.policy.model
            target_gat = target_config.policy.model
            
            # 重要なGAT設定が一致しているか確認
            critical_params = [
                'num_heads', 'hidden_channels', 'num_gat_layers', 
                'state_dim', 'action_space_size'
            ]
            
            for param in critical_params:
                if source_gat.get(param) != target_gat.get(param):
                    return False, f"GAT設定が不一致: {param} (ソース: {source_gat.get(param)}, ターゲット: {target_gat.get(param)})"
            
            # グリッドサイズの確認
            source_grid_size = self.extract_grid_size_from_config(source_config)
            target_grid_size = self.extract_grid_size_from_config(target_config)
            
            if source_grid_size >= target_grid_size:
                return False, f"転移学習は小さいグリッドから大きいグリッドへのみサポート (ソース: {source_grid_size}, ターゲット: {target_grid_size})"
            
            return True, f"転移学習互換性確認済み: {source_grid_size}×{source_grid_size} → {target_grid_size}×{target_grid_size}"
            
        except Exception as e:
            return False, f"互換性確認エラー: {e}"
    
    def create_transfer_experiment_name(self, source_exp_name: str, target_grid_size: int) -> str:
        """転移学習用の実験名を生成"""
        # ソース実験名からgrid情報を抽出
        import re
        grid_pattern = r'grid(\d+)'
        match = re.search(grid_pattern, source_exp_name)
        
        if match:
            source_grid_size = match.group(1)
            # ソースのgridサイズをターゲットのサイズに置き換え
            transfer_exp_name = re.sub(
                grid_pattern, 
                f'grid{target_grid_size}', 
                source_exp_name
            )
            # 転移学習であることを明示
            transfer_exp_name = transfer_exp_name.replace('data_gat', f'data_gat_transfer_{source_grid_size}to{target_grid_size}')
        else:
            # フォールバック
            transfer_exp_name = f"transfer_to_grid{target_grid_size}_{source_exp_name}"
        
        return transfer_exp_name


def create_transfer_config(source_config, target_grid_size: int, pretrained_model_path: str):
    """
    転移学習用の設定を作成
    
    Args:
        source_config: ソース設定
        target_grid_size: ターゲットグリッドサイズ
        pretrained_model_path: プリトレインモデルのパス
        
    Returns:
        転移学習用設定
    """
    # ソース設定をコピー
    transfer_config = source_config.copy()
    
    # ターゲットグリッドサイズに応じて設定を更新
    num_of_possible_chance_tile = transfer_config.env.get('num_of_possible_chance_tile', 2)
    target_chance_space_size = (target_grid_size ** 2) * num_of_possible_chance_tile
    
    # 環境設定の更新
    transfer_config.env.obs_shape = (16, target_grid_size, target_grid_size)
    transfer_config.env.num_of_possible_chance_tile = num_of_possible_chance_tile
    
    # モデル設定の更新
    transfer_config.policy.model.observation_shape = (16, target_grid_size, target_grid_size)
    transfer_config.policy.model.chance_space_size = target_chance_space_size
    
    # プリトレインモデルのパスを設定
    transfer_config.policy.model_path = pretrained_model_path
    
    # 転移学習用に学習率を調整（通常は下げる）
    if 'learning_rate' in transfer_config.policy:
        original_lr = transfer_config.policy.learning_rate
        transfer_config.policy.learning_rate = original_lr * 0.3  # 学習率を30%に削減
    
    # 実験名を更新
    helper = TransferLearningHelper()
    transfer_config.exp_name = helper.create_transfer_experiment_name(
        transfer_config.exp_name, target_grid_size
    )
    
    return transfer_config
