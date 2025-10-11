"""
CNN 2048 汎用サイズエミュレータ

4×4で学習したCNNモデルを任意の盤面サイズで動作させる独立したエミュレータです。
LightZeroのGUIに依存せず、コマンドラインで簡単に実行できます。

使い方:
    python cnn_any_size_emulator.py --model-path path/to/model.pth.tar
    python cnn_any_size_emulator.py --model-path path/to/model.pth.tar --episodes 10 --render
    python cnn_any_size_emulator.py --model-path path/to/model.pth.tar --save-gif

主な機能:
    - 4×4盤面でのプレイ
    - リアルタイム描画モード（--render）
    - GIFアニメーション保存（--save-gif）
    - 詳細な統計情報の表示
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import time

# LightZeroのパスを追加
sys.path.append('./LightZero')


# =============================================================================
# ゲーム環境（4×4版）
# =============================================================================

class Game2048:
    """4×4の2048ゲーム環境"""
    
    def __init__(self):
        """4×4固定"""
        self.grid_size = 4
        self.board = None
        self.score = 0
        self.max_tile = 0
        self.moves = 0
        self.reset()
    
    def reset(self):
        """ゲームをリセット"""
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.score = 0
        self.max_tile = 0
        self.moves = 0
        
        # 初期タイルを2つ配置
        self._add_random_tile()
        self._add_random_tile()
        
        return self._get_observation()
    
    def _add_random_tile(self):
        """ランダムな空きマスに新しいタイルを追加"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = empty_cells[np.random.randint(len(empty_cells))]
            # 90%の確率で2、10%の確率で4
            self.board[row, col] = 2 if np.random.random() < 0.9 else 4
    
    def _get_observation(self):
        """観測を取得（ワンホットエンコーディング形式）"""
        # 0から2048までのlog2値をエンコード（16チャンネル）
        obs = np.zeros((16, self.grid_size, self.grid_size), dtype=np.float32)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.board[i, j] > 0:
                    # log2(value)をチャンネルインデックスとして使用
                    channel = min(int(np.log2(self.board[i, j])), 15)
                    obs[channel, i, j] = 1.0
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        アクションを実行
        
        Args:
            action: 0=上, 1=右, 2=下, 3=左
        
        Returns:
            observation: 新しい観測
            reward: 報酬
            done: 終了フラグ
            info: 追加情報
        """
        old_board = self.board.copy()
        old_score = self.score
        
        # アクションを実行
        moved = self._move(action)
        
        # 移動が発生した場合、新しいタイルを追加
        if moved:
            self._add_random_tile()
            self.moves += 1
        
        # 報酬計算
        reward = self.score - old_score
        
        # 最大タイルを更新
        self.max_tile = np.max(self.board)
        
        # 終了判定
        done = not self._has_legal_moves()
        
        info = {
            'score': self.score,
            'max_tile': self.max_tile,
            'moves': self.moves,
            'legal_move': moved
        }
        
        return self._get_observation(), reward, done, info
    
    def _move(self, action: int) -> bool:
        """
        盤面を移動させる
        
        Returns:
            moved: 移動が発生したかどうか
        """
        old_board = self.board.copy()
        
        if action == 0:  # 上
            self._move_up()
        elif action == 1:  # 右
            self._move_right()
        elif action == 2:  # 下
            self._move_down()
        elif action == 3:  # 左
            self._move_left()
        
        return not np.array_equal(old_board, self.board)
    
    def _move_left(self):
        """左に移動"""
        for i in range(self.grid_size):
            self.board[i, :] = self._merge_line(self.board[i, :])
    
    def _move_right(self):
        """右に移動"""
        for i in range(self.grid_size):
            self.board[i, :] = self._merge_line(self.board[i, ::-1])[::-1]
    
    def _move_up(self):
        """上に移動"""
        self.board = self.board.T
        self._move_left()
        self.board = self.board.T
    
    def _move_down(self):
        """下に移動"""
        self.board = self.board.T
        self._move_right()
        self.board = self.board.T
    
    def _merge_line(self, line: np.ndarray) -> np.ndarray:
        """1行をマージ"""
        # 0を除去
        non_zero = line[line != 0]
        
        if len(non_zero) == 0:
            return line
        
        # マージ
        merged = []
        skip = False
        
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                # マージ
                merged_value = non_zero[i] * 2
                merged.append(merged_value)
                self.score += merged_value
                skip = True
            else:
                merged.append(non_zero[i])
        
        # 0で埋める
        result = np.zeros(self.grid_size, dtype=np.int32)
        result[:len(merged)] = merged
        
        return result
    
    def _has_legal_moves(self) -> bool:
        """合法手があるかチェック"""
        # 空きマスがある
        if np.any(self.board == 0):
            return True
        
        # 隣接するマスに同じ値がある
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                current = self.board[i, j]
                # 右
                if j + 1 < self.grid_size and self.board[i, j + 1] == current:
                    return True
                # 下
                if i + 1 < self.grid_size and self.board[i + 1, j] == current:
                    return True
        
        return False
    
    def get_legal_actions(self) -> List[int]:
        """合法なアクションのリストを取得"""
        legal_actions = []
        
        for action in range(4):
            old_board = self.board.copy()
            self._move(action)
            if not np.array_equal(old_board, self.board):
                legal_actions.append(action)
            self.board = old_board
        
        return legal_actions


# =============================================================================
# CNN モデル（LightZeroの標準MuZeroモデル）
# =============================================================================

class RepresentationNetwork(nn.Module):
    """CNN表現ネットワーク"""
    
    def __init__(
        self,
        observation_shape: Tuple[int, int, int] = (16, 4, 4),
        num_res_blocks: int = 1,
        num_channels: int = 64,
        downsample: bool = False,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.num_channels = num_channels
        
        # 入力層
        self.conv1 = nn.Conv2d(observation_shape[0], num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # Residual blocks
        self.resblocks = nn.ModuleList([
            ResidualBlock(num_channels, num_channels) for _ in range(num_res_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        
        Returns:
            latent_state: [B, num_channels, H, W]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        for block in self.resblocks:
            x = block(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual Block"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = F.relu(out)
        
        return out


class PolicyHead(nn.Module):
    """CNNポリシーヘッド"""
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        action_space_size: int,
    ):
        super().__init__()
        c, h, w = input_shape
        
        self.conv = nn.Conv2d(c, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        
        fc_input_dim = 2 * h * w
        self.fc = nn.Linear(fc_input_dim, action_space_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        
        Returns:
            policy_logits: [B, action_space_size]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class ValueHead(nn.Module):
    """CNNバリューヘッド"""
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        value_support_size: int,
        fc_value_layers: List[int] = [64],
    ):
        super().__init__()
        c, h, w = input_shape
        
        self.conv = nn.Conv2d(c, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        
        fc_input_dim = h * w
        
        # FC layers
        layers = []
        prev_dim = fc_input_dim
        for hidden_dim in fc_value_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, value_support_size))
        
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        
        Returns:
            value: [B, value_support_size]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class CNNAgent:
    """CNNエージェント"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Args:
            model_path: 学習済みモデルのパス
            device: デバイス
        """
        self.device = device
        
        # モデルを構築
        self._build_model()
        
        # モデルをロード
        self._load_model(model_path)
        
        # 評価モード
        self.representation_net.eval()
        self.policy_head.eval()
        self.value_head.eval()
    
    def _build_model(self):
        """モデルを構築"""
        self.representation_net = RepresentationNetwork(
            observation_shape=(16, 4, 4),
            num_res_blocks=1,
            num_channels=64,
            downsample=False,
        ).to(self.device)
        
        self.policy_head = PolicyHead(
            input_shape=(64, 4, 4),
            action_space_size=4,
        ).to(self.device)
        
        self.value_head = ValueHead(
            input_shape=(64, 4, 4),
            value_support_size=601,  # LightZeroのデフォルト
            fc_value_layers=[64],
        ).to(self.device)
    
    def _load_model(self, model_path: str):
        """学習済みモデルをロード"""
        print(f"モデルをロード中: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # state_dictを取得
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # キーの変換
        rep_state_dict = {}
        policy_state_dict = {}
        value_state_dict = {}
        
        for key, value in state_dict.items():
            if 'representation_network' in key:
                new_key = key.replace('representation_network.', '')
                rep_state_dict[new_key] = value
            elif 'prediction_network.policy_head' in key:
                new_key = key.replace('prediction_network.policy_head.', '')
                policy_state_dict[new_key] = value
            elif 'prediction_network.value_head' in key:
                new_key = key.replace('prediction_network.value_head.', '')
                value_state_dict[new_key] = value
        
        # ロード
        try:
            self.representation_net.load_state_dict(rep_state_dict, strict=False)
            self.policy_head.load_state_dict(policy_state_dict, strict=False)
            self.value_head.load_state_dict(value_state_dict, strict=False)
            print("✓ モデルのロード完了")
        except Exception as e:
            print(f"警告: モデルのロード中にエラーが発生しました: {e}")
            print("部分的にロードを試みます...")
            self.representation_net.load_state_dict(rep_state_dict, strict=False)
            self.policy_head.load_state_dict(policy_state_dict, strict=False)
            self.value_head.load_state_dict(value_state_dict, strict=False)
    
    def select_action(self, observation: np.ndarray, legal_actions: Optional[List[int]] = None) -> int:
        """
        アクションを選択
        
        Args:
            observation: 観測 [C, H, W]
            legal_actions: 合法アクションのリスト
        
        Returns:
            action: 選択されたアクション
        """
        with torch.no_grad():
            # 観測をテンソルに変換
            obs_tensor = torch.from_numpy(observation).unsqueeze(0).float().to(self.device)
            
            # 表現ネットワークで潜在状態を取得
            latent_state = self.representation_net(obs_tensor)
            
            # ポリシーヘッドでアクション確率を取得
            policy_logits = self.policy_head(latent_state)
            
            # 合法アクションでマスク
            if legal_actions is not None and len(legal_actions) > 0:
                mask = torch.full_like(policy_logits, float('-inf'))
                mask[0, legal_actions] = 0
                policy_logits = policy_logits + mask
            
            # 確率分布に変換
            policy_probs = F.softmax(policy_logits, dim=-1)
            
            # 最も確率の高いアクションを選択
            action = torch.argmax(policy_probs, dim=-1).item()
            
            return action


# =============================================================================
# 評価・可視化
# =============================================================================

def render_board_cli(board: np.ndarray, score: int, max_tile: int, moves: int):
    """盤面をCLIでテキスト表示"""
    grid_size = board.shape[0]
    
    print("\n" + "=" * (grid_size * 8 + 1))
    for row in board:
        print("|", end="")
        for cell in row:
            if cell == 0:
                print("      |", end="")
            else:
                print(f"{cell:6d}|", end="")
        print()
        print("-" * (grid_size * 8 + 1))
    
    print(f"スコア: {score} | 手数: {moves} | 最大タイル: {max_tile}")
    print("=" * (grid_size * 8 + 1))


def evaluate_model(agent: CNNAgent, num_episodes: int = 10, render: bool = False, save_gif: bool = False):
    """
    モデルを評価
    
    Args:
        agent: エージェント
        num_episodes: エピソード数
        render: CLIでリアルタイム表示するか
        save_gif: GIFを保存するか
    """
    env = Game2048()
    
    scores = []
    max_tiles = []
    moves_list = []
    
    # GIF保存用の準備
    if save_gif:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_frames = []
        
        print(f"\n{'='*60}")
        print(f"エピソード {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        step = 0
        while not done:
            # 合法アクションを取得
            legal_actions = env.get_legal_actions()
            
            if len(legal_actions) == 0:
                break
            
            # アクションを選択
            action = agent.select_action(obs, legal_actions)
            
            # 環境でステップ
            obs, reward, done, info = env.step(action)
            
            step += 1
            
            # CLI描画
            if render:
                render_board_cli(env.board, env.score, env.max_tile, step)
                time.sleep(0.1)
            
            if save_gif:
                episode_frames.append(env.board.copy())
        
        # 統計情報を記録
        scores.append(env.score)
        max_tiles.append(env.max_tile)
        moves_list.append(env.moves)
        
        print(f"\n最終結果:")
        print(f"  スコア: {env.score}")
        print(f"  最大タイル: {env.max_tile}")
        print(f"  手数: {env.moves}")
        
        # GIFを保存
        if save_gif and episode_frames:
            save_episode_gif(episode_frames, scores[-1], max_tiles[-1], episode)
    
    # 統計情報を表示
    print(f"\n{'='*60}")
    print("統計情報")
    print(f"{'='*60}")
    print(f"平均スコア: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"最高スコア: {np.max(scores)}")
    print(f"平均最大タイル: {np.mean(max_tiles):.1f}")
    print(f"最大タイル分布:")
    
    tile_counts = {}
    for tile in max_tiles:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    
    for tile in sorted(tile_counts.keys(), reverse=True):
        count = tile_counts[tile]
        percentage = count / len(max_tiles) * 100
        print(f"  {tile}: {count}回 ({percentage:.1f}%)")
    
    print(f"\n平均手数: {np.mean(moves_list):.1f}")
    print(f"{'='*60}")


def render_board_matplotlib(board: np.ndarray, score: int, max_tile: int, step: int, ax):
    """盤面をmatplotlibで描画（GIF保存用）"""
    ax.clear()
    
    # タイルの色を定義
    colors = {
        0: '#cdc1b4',
        2: '#eee4da',
        4: '#ede0c8',
        8: '#f2b179',
        16: '#f59563',
        32: '#f67c5f',
        64: '#f65e3b',
        128: '#edcf72',
        256: '#edcc61',
        512: '#edc850',
        1024: '#edc53f',
        2048: '#edc22e',
    }
    
    grid_size = board.shape[0]
    
    # 盤面を描画
    for i in range(grid_size):
        for j in range(grid_size):
            value = board[i, j]
            color = colors.get(value, '#3c3a32')
            
            rect = plt.Rectangle((j, grid_size - 1 - i), 1, 1, 
                                 facecolor=color, edgecolor='#bbada0', linewidth=2)
            ax.add_patch(rect)
            
            if value > 0:
                text_color = '#776e65' if value <= 4 else '#f9f6f2'
                ax.text(j + 0.5, grid_size - 1 - i + 0.5, str(value),
                       ha='center', va='center', fontsize=24 if value < 1000 else 20,
                       color=text_color, weight='bold')
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # スコアとステップを表示
    ax.set_title(f'Score: {score} | Max Tile: {max_tile} | Step: {step}', 
                fontsize=16, pad=20)


def save_episode_gif(frames: List[np.ndarray], score: int, max_tile: int, episode: int):
    """エピソードのGIFを保存"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(frame_idx):
        render_board_matplotlib(frames[frame_idx], score, max_tile, frame_idx + 1, ax)
    
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=200)
    
    filename = f'cnn_eval_ep{episode}_score{score}_tile{max_tile}.gif'
    anim.save(filename, writer='pillow')
    print(f"GIFを保存しました: {filename}")
    
    plt.close()


# =============================================================================
# メイン
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='CNN 2048 Emulator')
    parser.add_argument('--model-path', type=str, required=True,
                       help='学習済みモデルのパス')
    parser.add_argument('--episodes', type=int, default=10,
                       help='評価エピソード数 (default: 10)')
    parser.add_argument('--render', action='store_true',
                       help='リアルタイム描画を有効にする')
    parser.add_argument('--save-gif', action='store_true',
                       help='GIFアニメーションを保存する')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='デバイス (default: auto)')
    
    args = parser.parse_args()
    
    # デバイスを設定
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"{'='*50}")
    print("CNN 2048 Emulator")
    print(f"{'='*50}")
    print(f"モデルパス: {args.model_path}")
    print(f"エピソード数: {args.episodes}")
    print(f"デバイス: {device}")
    print(f"{'='*50}\n")
    
    # エージェントを作成
    agent = CNNAgent(
        model_path=args.model_path,
        device=device,
    )
    
    # 評価を実行
    evaluate_model(
        agent=agent,
        num_episodes=args.episodes,
        render=args.render,
        save_gif=args.save_gif,
    )


if __name__ == '__main__':
    main()
