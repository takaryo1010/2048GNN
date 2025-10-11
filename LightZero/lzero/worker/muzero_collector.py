import os
import time
from collections import deque, namedtuple
from typing import Optional, Any, List, TYPE_CHECKING

import numpy as np
import torch
import wandb
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY, get_rank, get_world_size, \
    allreduce_data
from ding.worker.collector.base_serial_collector import ISerialCollector
from torch.nn import L1Loss
import torch.distributed as dist

from lzero.mcts.buffer.game_segment import GameSegment
from lzero.mcts.utils import prepare_observation
from lzero.policy.utils import compute_bleu

if TYPE_CHECKING:
    # 型チェック時のみインポート（実行時には不要）
    from tensorboardX import SummaryWriter  # type: ignore
    # policy_config の具体的な型は実行時には必ずしも必要ないため TYPE_CHECKING 時のみ参照
    from typing import Any as policy_config  # type: ignore


@SERIAL_COLLECTOR_REGISTRY.register('episode_muzero')
class MuZeroCollector(ISerialCollector):
    """
    概要:
        MCTS+RL（MuZero, EfficientZero 等）向けのエピソード収集クラス（シリアル実行版）。
        環境との相互作用を管理し、学習用のゲームセグメントを生成してプールに保存します。
    インターフェース:
        __init__, reset, reset_env, reset_policy, _reset_stat, envstep, __del__, _compute_priorities,
        pad_and_save_last_trajectory, collect, _output_log, close
    プロパティ:
        envstep: 収集した総環境ステップ数
    """

    # TO be compatible with ISerialCollector
    config = dict()

    def __init__(
            self,
            collect_print_freq: int = 100,
            env: BaseEnvManager = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'collector',
            policy_config: 'policy_config' = None,  # noqa
    ) -> None:
        """
        概要:
            MuZeroCollector を初期化します。
        引数:
            collect_print_freq: ログ出力の頻度（学習イテレーション単位）
            env: ベクトル化環境マネージャのインスタンス（省略可）
            policy: collect_mode 用ポリシーの namedtuple（API を持つこと）
            tb_logger: TensorBoard 用ロガー（ランク 0 のみ使用）
            exp_name: 実験名（ログ/チェックポイント保存に利用）
            instance_name: コレクタの識別名
            policy_config: ポリシー設定オブジェクト（各種フラグやサイズなどを含む）
        """
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._collect_print_freq = collect_print_freq
        self._timer = EasyTimer()
        self._end_flag = False

        self._rank = get_rank()
        self._world_size = get_world_size()
        if self._rank == 0:
            if tb_logger is not None:
                self._logger, _ = build_logger(
                    path='./{}/log/{}'.format(self._exp_name, self._instance_name),
                    name=self._instance_name,
                    need_tb=False
                )
                self._tb_logger = tb_logger
            else:
                self._logger, self._tb_logger = build_logger(
                    path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name
                )
        else:
            self._logger, _ = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name, need_tb=False
            )
            self._tb_logger = None

        # ポリシー設定を保存
        self.policy_config = policy_config
        # collect_with_pure_policy フラグをキャッシュ（ポリシー設定から取得）
        self.collect_with_pure_policy = self.policy_config.collect_with_pure_policy

        # ポリシーと環境の初期化処理を呼ぶ（与えられていれば置換、それ以外はリセット）
        self.reset(policy, env)

    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        """
        概要:
            コレクタが管理する環境をリセットまたは置換します。
            _env が None の場合は既存環境をリセットし、そうでなければ新しい環境に置換して起動します。
        引数:
            _env: 置換する新しい BaseEnvManager（省略可）
        """
        if _env is not None:
            self._env = _env
            self._env.launch()
            self._env_num = self._env.env_num
        else:
            self._env.reset()

    def reset_policy(self, _policy: Optional[namedtuple] = None) -> None:
        """
        概要:
            コレクタが使用するポリシーをリセットまたは置換します。
            _policy が None の場合は既存ポリシーをリセットし、そうでなければ新しいポリシーを設定します。
        引数:
            _policy: collect_mode 用ポリシーの namedtuple（省略可）
        """
        assert hasattr(self, '_env'), "please set env first"
        if _policy is not None:
            self._policy = _policy
            self._default_n_episode = _policy.get_attribute('cfg').get('n_episode', None)
            self._logger.debug(
                'Set default n_episode mode(n_episode({}), env_num({}))'.format(self._default_n_episode, self._env_num)
            )
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        概要:
            コレクタ全体をリセットまたはポリシー/環境を置換します。
            引数に応じて env/policy の置換または既存オブジェクトのリセットを行います。
        引数:
            _policy: collect_mode ポリシー（省略可）
            _env: BaseEnvManager インスタンス（省略可）
        """
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)

        # 環境ごとのメタ情報（時間、ステップ、text BLEU 等）を初期化
        self._env_info = {env_id: {'time': 0., 'step': 0, 'text_bleu': 0.} for env_id in range(self._env_num)}

        # ログ用エピソード情報のリスト
        self._episode_info = []
        # 累積した環境ステップ数
        self._total_envstep_count = 0
        # 累積したエピソード数
        self._total_episode_count = 0
        # 累積実行時間
        self._total_duration = 0
        # 最後にログ出力した学習イテレーション
        self._last_train_iter = 0
        # 終了フラグ
        self._end_flag = False

        # ゲームセグメントのプール（deque を使用して固定長で保持）
        self.game_segment_pool = deque(maxlen=int(1e6))
        # unroll ステップ数 + td ステップ数 の合計をキャッシュ
        self.unroll_plus_td_steps = self.policy_config.num_unroll_steps + self.policy_config.td_steps

    def _reset_stat(self, env_id: int) -> None:
        """
        概要:
            指定した env_id に対して collector の統計情報（env_info 等）をリセットします。
        引数:
            env_id: リセット対象の環境 ID
        """
        self._env_info[env_id] = {'time': 0., 'step': 0, 'text_bleu': 0.}

    @property
    def envstep(self) -> int:
        """
        概要:
            これまでに収集した総環境ステップ数を返します。
        戻り値:
            envstep: int 型の総ステップ数
        """
        return self._total_envstep_count

    def close(self) -> None:
        """
        概要:
            コレクタを終了します。まだ終了していなければ環境を閉じ、TensorBoard ロガーをフラッシュして閉じます。
        """
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        if self._tb_logger:
            self._tb_logger.flush()
            self._tb_logger.close()

    def __del__(self) -> None:
        """
        概要:
            オブジェクト削除時に close() を実行してクリーンアップします。
        """
        self.close()

    # ==============================================================
    # MCTS+RL related core code
    # ==============================================================
    def _compute_priorities(self, i: int, pred_values_lst: List[float], search_values_lst: List[float]) -> np.ndarray:
        """
        概要:
            予測値（pred）と探索値（search）の差から、優先度（priorities）を計算します。
        引数:
            i: リストのインデックス（どの遷移に対して計算するか）
            pred_values_lst: 各ステップの予測値リスト（各 env ごと）
            search_values_lst: 各ステップの MCTS による検索値リスト
        戻り値:
            priorities: numpy 配列。優先度が有効でない場合は None を返す。
        """
        if self.policy_config.use_priority:
            # 優先度を L1 誤差として計算する（各要素ごとの誤差を保持するため reduction='none' を利用）
            # small constant を足して 0 優先度を避ける
            pred_values = torch.from_numpy(np.array(pred_values_lst[i])).to(self.policy_config.device).float().view(-1)
            search_values = torch.from_numpy(np.array(search_values_lst[i])).to(self.policy_config.device).float().view(-1)
            priorities = L1Loss(reduction='none')(pred_values, search_values).detach().cpu().numpy() + 1e-6
        else:
            # 優先度を使わない場合は None を返し、外側でデフォルトの扱い（最大優先度付与）を行う
            priorities = None

        return priorities

    def pad_and_save_last_trajectory(self, i: int, last_game_segments: List[GameSegment],
                                     last_game_priorities: List[np.ndarray],
                                     game_segments: List[GameSegment], done: np.ndarray) -> None:
        """
        概要:
            最後のゲームセグメントを必要に応じてパディングして保存します。
        引数:
            i: 現在処理している環境インデックス
            last_game_segments: 前のセグメント（パディング対象）リスト
            last_game_priorities: 前セグメントの優先度リスト
            game_segments: 現在のセグメントリスト
            done: 各環境の終了フラグ配列

        注意:
            last_game_segments[i].obs_segment[-frame_stack_num:] と game_segments[i].obs_segment[:frame_stack_num] の重複を前提としている
        """
        # obs のパッド用インデックスは frame_stack_num（先頭はゼロ埋め）から始める
        beg_index = self.policy_config.model.frame_stack_num
        end_index = beg_index + self.policy_config.num_unroll_steps + self.policy_config.td_steps

        # 先頭のフレームは初期ゼロ観測なので、その次のスライスをパッド用 obs とする
        pad_obs_lst = game_segments[i].obs_segment[beg_index:end_index]

        # UniZero 系では beg_index=0 から取る
        beg_index = 0
        end_index = beg_index + self.policy_config.num_unroll_steps + self.policy_config.td_steps
        pad_action_lst = game_segments[i].action_segment[beg_index:end_index]

        # child visits も同様にパッドを用意（unizero との互換性コメント）
        pad_child_visits_lst = game_segments[i].child_visit_segment[:self.policy_config.num_unroll_steps + self.policy_config.td_steps]

        # reward のパディング範囲
        beg_index = 0
        end_index = beg_index + self.unroll_plus_td_steps - 1
        pad_reward_lst = game_segments[i].reward_segment[beg_index:end_index]
        if self.policy_config.use_ture_chance_label_in_chance_encoder:
            # chance ラベルを使う場合は chance もパディング
            chance_lst = game_segments[i].chance_segment[beg_index:end_index]

        # root value のパディング
        beg_index = 0
        end_index = beg_index + self.unroll_plus_td_steps
        pad_root_values_lst = game_segments[i].root_value_segment[beg_index:end_index]

        # gumbel アルゴリズム用に improved policy をパッド
        if self.policy_config.gumbel_algo:
            pad_improved_policy_prob = game_segments[i].improved_policy_probs[beg_index:end_index]

        # 実際に last_game_segments[i] にパッドを適用して保存する
        if self.policy_config.gumbel_algo:
            last_game_segments[i].pad_over(pad_obs_lst, pad_reward_lst, pad_action_lst, pad_root_values_lst, pad_child_visits_lst,
                                           next_segment_improved_policy=pad_improved_policy_prob)
        else:
            if self.policy_config.use_ture_chance_label_in_chance_encoder:
                last_game_segments[i].pad_over(pad_obs_lst, pad_reward_lst, pad_action_lst, pad_root_values_lst, pad_child_visits_lst,
                                               next_chances=chance_lst)
            else:
                last_game_segments[i].pad_over(pad_obs_lst, pad_reward_lst, pad_action_lst, pad_root_values_lst, pad_child_visits_lst)

        """
        補足（game_segment 要素の形状イメージ）:
            obs: game_segment_length + stack + num_unroll_steps  (例: 20 + 4 + 5)
            rew: game_segment_length + stack + num_unroll_steps + td_steps - 1
            action: game_segment_length
            root_values: game_segment_length + num_unroll_steps + td_steps
            child_visits: game_segment_length + num_unroll_steps
            to_play: game_segment_length
            action_mask: game_segment_length
        """

        # array 形式に変換してからゲームセグメントプールへ追加
        last_game_segments[i].game_segment_to_array()
        self.game_segment_pool.append((last_game_segments[i], last_game_priorities[i], done[i]))

        # 参照をクリアして GC を促す
        last_game_segments[i] = None
        last_game_priorities[i] = None

        return None

    def collect(self,
                n_episode: Optional[int] = None,
                train_iter: int = 0,
                policy_kwargs: Optional[dict] = None,
                collect_with_pure_policy: bool = False) -> List[Any]:
        """
        概要:
            指定したエピソード数分だけデータを収集し、ゲームセグメントとメタ情報を返します。
        引数:
            n_episode: 収集するエピソード数（None の場合は既定値を使用）
            train_iter: これまでの学習イテレーション数（ログ用途）
            policy_kwargs: ポリシーに渡す追加引数（temperature, epsilon など）
            collect_with_pure_policy: MCTS を使わずピュアなポリシーで収集するかのフラグ
        戻り値:
            return_data: [game_segment_list, meta_data_list] のタプル形式のリスト
        """
        # TODO: collect_with_pure_policy を別コレクタに分離する検討
        if n_episode is None:
            if self._default_n_episode is None:
                raise RuntimeError("Please specify collect n_episode")
            else:
                n_episode = self._default_n_episode
        assert n_episode >= self._env_num, "Please make sure n_episode >= env_num{}/{}".format(n_episode, self._env_num)
        if policy_kwargs is None:
            policy_kwargs = {}
        # policy_kwargs から温度・ε を取得（収集時の探索強度や ε-greedy 用）
        temperature = policy_kwargs['temperature']
        epsilon = policy_kwargs['epsilon']

        # 収集中の統計変数を初期化
        collected_episode = 0
        collected_step = 0
        env_nums = self._env_num
        # サブプロセス環境待ちのリトライ間隔（秒）
        retry_waiting_time = 0.05

        # 初期化: ready_obs（各サブ環境の初期観測）が揃うまで待つ
        init_obs = self._env.ready_obs
        while len(init_obs.keys()) != self._env_num:
            # subprocess ベースの env_manager では ready_obs のキー数が _env_num と一致しない場合がある
            self._logger.info('The current init_obs.keys() is {}'.format(init_obs.keys()))
            self._logger.info('Before sleeping, the _env_states is {}'.format(self._env._env_states))
            time.sleep(retry_waiting_time)
            self._logger.info('=' * 10 + 'Wait for all environments (subprocess) to finish resetting.' + '=' * 10)
            self._logger.info('After sleeping {}s, the current _env_states is {}'.format(retry_waiting_time, self._env._env_states))
            init_obs = self._env.ready_obs

        # action mask / to_play を ndarray に変換して辞書化しておく
        action_mask_dict = {i: to_ndarray(init_obs[i]['action_mask']) for i in range(env_nums)}
        to_play_dict = {i: to_ndarray(init_obs[i]['to_play']) for i in range(env_nums)}

        # timestep は存在しない場合もあるため、デフォルト -1 を使用
        timestep_dict = {}
        for i in range(env_nums):
            if 'timestep' not in init_obs[i]:
                if self._policy.get_attribute('cfg').type in ['unizero', 'sampled_unizero']:
                    print(f"Warning: 'timestep' key is missing in init_obs[{i}]. Assigning value -1. Please note that the unizero algorithm may require the 'timestep' key in init_obs.")
            timestep_dict[i] = to_ndarray(init_obs[i].get('timestep', -1))

        # chance ラベルを使う設定なら初期値を準備
        if self.policy_config.use_ture_chance_label_in_chance_encoder:
            chance_dict = {i: to_ndarray(init_obs[i]['chance']) for i in range(env_nums)}

        # 各環境に対して GameSegment インスタンスを作成（観測窓を保持するため）
        game_segments = [
            GameSegment(
                self._env.action_space,
                game_segment_length=self.policy_config.game_segment_length,
                config=self.policy_config
            ) for _ in range(env_nums)
        ]
        # reset 時点のスタックされた観測ウィンドウを保持する deque を作る
        observation_window_stack = [[] for _ in range(env_nums)]
        for env_id in range(env_nums):
            observation_window_stack[env_id] = deque(
                [to_ndarray(init_obs[env_id]['observation']) for _ in range(self.policy_config.model.frame_stack_num)],
                maxlen=self.policy_config.model.frame_stack_num
            )
            game_segments[env_id].reset(observation_window_stack[env_id])

        # 終了フラグ配列や last_game_segments 等を初期化
        dones = np.array([False for _ in range(env_nums)])
        last_game_segments = [None for _ in range(env_nums)]
        last_game_priorities = [None for _ in range(env_nums)]
        # 優先度計算用の一時リスト
        search_values_lst = [[] for _ in range(env_nums)]
        pred_values_lst = [[] for _ in range(env_nums)]
        if self.policy_config.gumbel_algo:
            improved_policy_lst = [[] for _ in range(env_nums)]

        # ログ用の統計変数
        eps_steps_lst, visit_entropies_lst = np.zeros(env_nums), np.zeros(env_nums)
        if self.policy_config.gumbel_algo:
            completed_value_lst = np.zeros(env_nums)
        self_play_moves = 0.
        self_play_episodes = 0.
        self_play_moves_max = 0
        self_play_visit_entropy = []
        total_transitions = 0

        ready_env_id = set()
        remain_episode = n_episode
        if collect_with_pure_policy:
            # ピュアポリシー収集用の一時 visit リスト
            temp_visit_list = [0.0 for i in range(self._env.action_space.n)]

        while True:
            with self._timer:
                # Get current ready env obs.
                obs = self._env.ready_obs

                new_available_env_id = set(obs.keys()).difference(ready_env_id)
                ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                remain_episode -= min(len(new_available_env_id), remain_episode)

                # NOTE: If waiting for N environments to synchronize, it may result in some environments not being completed (done) by the time of return.
                # However, the current muzero_collector does not properly maintain the global self.last_game_segments, leading to some data not being collected.

                # ready_env_id の環境から観測を積み上げた stack_obs を作る
                stack_obs = {env_id: game_segments[env_id].get_obs() for env_id in ready_env_id}
                stack_obs = list(stack_obs.values())

                # 各種辞書から ready_env_id に相当する要素のみ取り出す
                action_mask_dict = {env_id: action_mask_dict[env_id] for env_id in ready_env_id}
                to_play_dict = {env_id: to_play_dict[env_id] for env_id in ready_env_id}
                timestep_dict = {env_id: timestep_dict[env_id] for env_id in ready_env_id}

                # バッチ化のためにリスト化
                action_mask = [action_mask_dict[env_id] for env_id in ready_env_id]
                to_play = [to_play_dict[env_id] for env_id in ready_env_id]
                timestep = [timestep_dict[env_id] for env_id in ready_env_id]

                if self.policy_config.use_ture_chance_label_in_chance_encoder:
                    chance_dict = {env_id: chance_dict[env_id] for env_id in ready_env_id}

                # numpy -> ネットワーク入力形式に整形 -> torch.Tensor に変換
                stack_obs = to_ndarray(stack_obs)
                # stack_obs の形状例: [B, S*C, W, H] e.g. [8, 4*1, 96, 96]
                stack_obs = prepare_observation(stack_obs, self.policy_config.model.model_type)
                stack_obs = torch.from_numpy(stack_obs).to(self.policy_config.device)

                # ==============================================================
                # ポリシーの前方伝播（MCTS の実行を含むことがある）
                # ==============================================================
                # 日本語注釈:
                # ここで collector はポリシーの `forward` を呼び出します。渡す引数は以下の通りです:
                # - stack_obs: バッチ化された観測テンソル (torch.Tensor), 形状はモデルに依存します（GNN の場合はノード表現など）
                # - action_mask: 各環境ごとの行動可能マスクのリスト（各要素は ndarray）
                # - temperature: MCTS のサンプリング温度（float）。大きいほど探索的。
                # - to_play: 各環境の現在のプレイヤー情報（list）
                # - epsilon: 収集時の ε-greedy 用パラメータ（float）
                # - ready_env_id (kwarg): 今回バッチ化された環境 ID の集合（set または list）
                # - timestep (kwarg): 各環境の現在 timestep（リスト）
                #
                # 戻り値（policy_output）の期待フォーマット（collector が以下で参照）:
                # policy_output は dict で、キーは env_id、値は各種情報を含む dict。
                # 典型的なキー:
                # - 'action': 選択した行動（整数または配列）
                # - 'visit_count_distributions': MCTS の訪問回数分布（リスト）
                # - 'visit_count_distribution_entropy': 分布のエントロピー（float）
                # - 'searched_value': MCTS による評価値（float）
                # - 'predicted_value': ネットワークが予測した価値（float）
                # - 'predicted_next_text' (任意): テキスト予測がある場合の文字列
                # - 'timestep' (任意): ポリシーが返す timestep
                # collector はこれらを取り出して game_segment に保存します。
                
                # #どのポリシーを出力しているかデバッグ出力
                # self._logger.info(f"Using policy: {self._policy}")
                # self._logger.info(f"stack_obs shape: {stack_obs.shape}, action_mask length: {len(action_mask)}, to_play length: {len(to_play)}, timestep length: {len(timestep)}")
                # self._logger.info(f"ready_env_id: {ready_env_id}")
                # self._logger.info(f"chance_dict: {chance_dict}")
                policy_output = self._policy.forward(stack_obs, action_mask, temperature, to_play, epsilon, ready_env_id=ready_env_id, timestep=timestep)
                # --- デバッグログ（1回だけ）: stack_obs と policy_output の形状を出力 ---
                try:
                    if not hasattr(self, '_debug_logged_policy_shapes'):
                        # stack_obs は torch.Tensor のはず
                        obs_shape = tuple(stack_obs.shape) if hasattr(stack_obs, 'shape') else str(type(stack_obs))
                        # policy_output は dict: env_id -> dict の構造であるため、最初の要素で shape を確認
                        first_key = next(iter(policy_output)) if len(policy_output) > 0 else None
                        if first_key is not None:
                            sample_out = policy_output[first_key]
                            action_shape = getattr(sample_out.get('action'), 'shape', str(type(sample_out.get('action'))))
                            searched_value_shape = getattr(sample_out.get('searched_value'), 'shape', str(type(sample_out.get('searched_value'))))
                            predicted_value_shape = getattr(sample_out.get('predicted_value'), 'shape', str(type(sample_out.get('predicted_value'))))
                        else:
                            action_shape = searched_value_shape = predicted_value_shape = 'empty'

                        self._logger.info(
                            'Debug collect shapes: stack_obs=%s, action=%s, searched_value=%s, predicted_value=%s',
                            obs_shape, action_shape, searched_value_shape, predicted_value_shape
                        )
                        # フラグを立てて一度だけログする
                        self._debug_logged_policy_shapes = True
                except Exception:
                    # デバッグログは補助的なものなので例外が発生しても無視する
                    try:
                        self._logger.exception('Failed to log policy shapes in collector')
                    except Exception:
                        pass

                # テキスト予測があれば取り出す（無ければ -1 を入れておく）
                pred_next_text_with_env_id = {k: v['predicted_next_text'] if 'predicted_next_text' in v else -1 for k, v in policy_output.items()}

                # 必要な出力のみを抜き出す
                actions_with_env_id = {k: v['action'] for k, v in policy_output.items()}
                value_dict_with_env_id = {k: v['searched_value'] for k, v in policy_output.items()}
                pred_value_dict_with_env_id = {k: v['predicted_value'] for k, v in policy_output.items()}
                timestep_dict_with_env_id = {k: v['timestep'] if 'timestep' in v else -1 for k, v in policy_output.items()}

                if self.policy_config.sampled_algo:
                    root_sampled_actions_dict_with_env_id = {k: v['root_sampled_actions'] for k, v in policy_output.items()}

                if not collect_with_pure_policy:
                    distributions_dict_with_env_id = {k: v['visit_count_distributions'] for k, v in policy_output.items()}
                    visit_entropy_dict_with_env_id = {k: v['visit_count_distribution_entropy'] for k, v in policy_output.items()}

                    if self.policy_config.gumbel_algo:
                        improved_policy_dict_with_env_id = {k: v['improved_policy_probs'] for k, v in policy_output.items()}
                        completed_value_with_env_id = {k: v['roots_completed_value'] for k, v in policy_output.items()}

                # 出力格納用辞書を初期化
                actions = {}
                value_dict = {}
                pred_value_dict = {}
                timestep_dict = {}
                pred_next_text = {}

                if not collect_with_pure_policy:
                    distributions_dict = {}
                    visit_entropy_dict = {}

                    if self.policy_config.sampled_algo:
                        root_sampled_actions_dict = {}

                    if self.policy_config.gumbel_algo:
                        improved_policy_dict = {}
                        completed_value_dict = {}

                # ready_env_id に対して出力辞書から要素を取り出して結果辞書を作る
                for env_id in ready_env_id:
                    actions[env_id] = actions_with_env_id.pop(env_id)
                    value_dict[env_id] = value_dict_with_env_id.pop(env_id)
                    pred_value_dict[env_id] = pred_value_dict_with_env_id.pop(env_id)
                    timestep_dict[env_id] = timestep_dict_with_env_id.pop(env_id)
                    pred_next_text[env_id] = pred_next_text_with_env_id.pop(env_id)

                    if not collect_with_pure_policy:
                        distributions_dict[env_id] = distributions_dict_with_env_id.pop(env_id)

                        if self.policy_config.sampled_algo:
                            root_sampled_actions_dict[env_id] = root_sampled_actions_dict_with_env_id.pop(env_id)

                        visit_entropy_dict[env_id] = visit_entropy_dict_with_env_id.pop(env_id)

                        if self.policy_config.gumbel_algo:
                            improved_policy_dict[env_id] = improved_policy_dict_with_env_id.pop(env_id)
                            completed_value_dict[env_id] = completed_value_with_env_id.pop(env_id)
        
                # ==============================================================
                # 環境と相互作用（Action を渡して step を取得）
                # ==============================================================
                timesteps = self._env.step(actions)

            # 各相互作用の平均所要時間を計算
            interaction_duration = self._timer.value / len(timesteps)
            
                    # 各環境の timestep に対する処理ループ
            groundtrut_next_text = {}
            for env_id, episode_timestep in timesteps.items():
                with self._timer:
                    # 異常なステップがあればその環境をリセットしてスキップ
                    if episode_timestep.info.get('abnormal', False):
                        self._env.reset({env_id: None})
                        self._policy.reset([env_id])
                        self._reset_stat(env_id)
                        self._logger.info('Env{} returns a abnormal step, its info is {}'.format(env_id, episode_timestep.info))
                        continue

                    # 通常の obs/reward/done/info を取得
                    obs, reward, done, info = episode_timestep.obs, episode_timestep.reward, episode_timestep.done, episode_timestep.info

                    # テキスト観測を使う world model の場合はデコードして BLEU を計算
                    if "world_model_cfg" in self.policy_config.model and self.policy_config.model.world_model_cfg.obs_type == 'text':
                        obs_input_ids = torch.tensor(obs['observation'], dtype=torch.long)  # shape: [L]
                        obs_attn_mask = torch.tensor(obs['obs_attn_mask'][0], dtype=torch.long)
                        valid_input_ids = obs_input_ids[obs_attn_mask == 1].tolist()

                        groundtrut_next_text[env_id] = self._env._envs[env_id].tokenizer.decode(valid_input_ids, skip_special_tokens=True)
                        text_bleu = compute_bleu(reference=groundtrut_next_text[env_id], prediction=pred_next_text[env_id])
                        # BLEU が高いものはログに保存しておく
                        if text_bleu > 0.85:
                            os.makedirs("./log", exist_ok=True)
                            with open("./log/bleu_match.txt", "a", encoding="utf-8") as f:
                                f.write(f"pred_text={pred_next_text[env_id]}\ngroundtruth_text={groundtrut_next_text[env_id]}\ntext_bleu={text_bleu:.4f}\n\n")

                    # store_search_stats に渡すデータを準備（pure policy 時は固定リストを入れる）
                    if collect_with_pure_policy:
                        game_segments[env_id].store_search_stats(temp_visit_list, 0)
                    else:
                        if self.policy_config.sampled_algo:
                            game_segments[env_id].store_search_stats(distributions_dict[env_id], value_dict[env_id], root_sampled_actions_dict[env_id])
                        elif self.policy_config.gumbel_algo:
                            game_segments[env_id].store_search_stats(distributions_dict[env_id], value_dict[env_id], improved_policy=improved_policy_dict[env_id])
                        else:
                            game_segments[env_id].store_search_stats(distributions_dict[env_id], value_dict[env_id])

                    # 遷移情報を append（action, next_obs, reward, action_mask, to_play 等）
                    if self.policy_config.use_ture_chance_label_in_chance_encoder:
                        game_segments[env_id].append(actions[env_id], to_ndarray(obs['observation']), reward, action_mask_dict[env_id], to_play_dict[env_id], timestep_dict[env_id], chance_dict[env_id])
                    else:
                        game_segments[env_id].append(actions[env_id], to_ndarray(obs['observation']), reward, action_mask_dict[env_id], to_play_dict[env_id], timestep_dict[env_id])

                    # 注意: ここで action_mask / to_play を次 timestep 用に更新する
                    action_mask_dict[env_id] = to_ndarray(obs['action_mask'])
                    to_play_dict[env_id] = to_ndarray(obs['to_play'])
                    timestep_dict[env_id] = to_ndarray(obs.get('timestep', -1))
                    if self.policy_config.use_ture_chance_label_in_chance_encoder:
                        chance_dict[env_id] = to_ndarray(obs['chance'])

                    # done を無視する設定なら False 固定
                    if self.policy_config.ignore_done:
                        dones[env_id] = False
                    else:
                        dones[env_id] = done

                    # ログ用統計の更新
                    if not collect_with_pure_policy:
                        visit_entropies_lst[env_id] += visit_entropy_dict[env_id]
                        if self.policy_config.gumbel_algo:
                            completed_value_lst[env_id] += np.mean(np.array(completed_value_dict[env_id]))

                    eps_steps_lst[env_id] += 1
                    if self._policy.get_attribute('cfg').type in ['unizero', 'sampled_unizero']:
                        # UniZero 特有のリセット処理（初期データは保持）
                        self._policy.reset(env_id=env_id, current_steps=eps_steps_lst[env_id], reset_init_data=False)

                    total_transitions += 1

                    # 優先度学習用に pred/search 値を蓄積
                    if self.policy_config.use_priority:
                        pred_values_lst[env_id].append(pred_value_dict[env_id])
                        search_values_lst[env_id].append(value_dict[env_id])
                        if self.policy_config.gumbel_algo and not collect_with_pure_policy:
                            improved_policy_lst[env_id].append(improved_policy_dict[env_id])

                    # 最新の観測を観測ウィンドウに追加
                    observation_window_stack[env_id].append(to_ndarray(obs['observation']))

                    # ==============================================================
                    # ゲームセグメントが満杯なら最後のセグメントをパッドして保存
                    # ==============================================================

                    if game_segments[env_id].is_full():
                        # 直前の last_game_segments があればそれを pad してプールに保存
                        if last_game_segments[env_id] is not None:
                            self.pad_and_save_last_trajectory(env_id, last_game_segments, last_game_priorities, game_segments, dones)

                        # 優先度を計算してリセット
                        priorities = self._compute_priorities(env_id, pred_values_lst, search_values_lst)
                        pred_values_lst[env_id] = []
                        search_values_lst[env_id] = []
                        if self.policy_config.gumbel_algo and not collect_with_pure_policy:
                            improved_policy_lst[env_id] = []

                        # 現在の game_segments を last_game_segments として交換
                        last_game_segments[env_id] = game_segments[env_id]
                        last_game_priorities[env_id] = priorities

                        # 新しい GameSegment を生成して初期化
                        game_segments[env_id] = GameSegment(self._env.action_space, game_segment_length=self.policy_config.game_segment_length, config=self.policy_config)
                        game_segments[env_id].reset(observation_window_stack[env_id])

                    # 環境ごとのステップ数をインクリメント
                    self._env_info[env_id]['step'] += 1
                    if "world_model_cfg" in self.policy_config.model and self.policy_config.model.world_model_cfg.obs_type == 'text':
                        self._env_info[env_id]['text_bleu'] += text_bleu

                    collected_step += 1

                # 1 ステップ分の経過時間を env_info に記録
                self._env_info[env_id]['time'] += self._timer.value + interaction_duration
                if episode_timestep.done:
                    # エピソード終了時のメタ情報収集
                    reward = episode_timestep.info['eval_episode_return']
                    info = {'reward': reward, 'time': self._env_info[env_id]['time'], 'step': self._env_info[env_id]['step']}
                    if "world_model_cfg" in self.policy_config.model and self.policy_config.model.world_model_cfg.obs_type == 'text':
                        info.update({'text_bleu': self._env_info[env_id]['text_bleu'] / self._env_info[env_id]['step']})

                    if not collect_with_pure_policy:
                        info['visit_entropy'] = visit_entropies_lst[env_id] / eps_steps_lst[env_id]
                        if self.policy_config.gumbel_algo:
                            info['completed_value'] = completed_value_lst[env_id] / eps_steps_lst[env_id]

                    collected_episode += 1
                    self._episode_info.append(info)

                    # ==============================================================
                    # エピソード終端: 残っているセグメントをプールへ保存
                    # ==============================================================

                    if last_game_segments[env_id] is not None:
                        self.pad_and_save_last_trajectory(env_id, last_game_segments, last_game_priorities, game_segments, dones)

                    # 現在のセグメントを配列に変換してプールに保存（優先度も計算）
                    priorities = self._compute_priorities(env_id, pred_values_lst, search_values_lst)
                    game_segments[env_id].game_segment_to_array()
                    if len(game_segments[env_id].reward_segment) != 0:
                        self.game_segment_pool.append((game_segments[env_id], priorities, dones[env_id]))

                    # n_episode > env_num の場合は環境を再初期化して次のエピソードを開始
                    if n_episode > self._env_num:
                        init_obs = self._env.ready_obs
                        retry_waiting_time = 0.001
                        while len(init_obs.keys()) != self._env_num:
                            self._logger.info('The current init_obs.keys() is {}'.format(init_obs.keys()))
                            self._logger.info('Before sleeping, the _env_states is {}'.format(self._env._env_states))
                            time.sleep(retry_waiting_time)
                            self._logger.info('=' * 10 + 'Wait for all environments (subprocess) to finish resetting.' + '=' * 10)
                            self._logger.info('After sleeping {}s, the current _env_states is {}'.format(retry_waiting_time, self._env._env_states))
                            init_obs = self._env.ready_obs

                        new_available_env_id = set(init_obs.keys()).difference(ready_env_id)
                        ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                        remain_episode -= min(len(new_available_env_id), remain_episode)

                        action_mask_dict[env_id] = to_ndarray(init_obs[env_id]['action_mask'])
                        to_play_dict[env_id] = to_ndarray(init_obs[env_id]['to_play'])
                        timestep_dict[env_id] = to_ndarray(init_obs[env_id].get('timestep', -1))

                        if self.policy_config.use_ture_chance_label_in_chance_encoder:
                            chance_dict[env_id] = to_ndarray(init_obs[env_id]['chance'])

                        game_segments[env_id] = GameSegment(self._env.action_space, game_segment_length=self.policy_config.game_segment_length, config=self.policy_config)
                        observation_window_stack[env_id] = deque([init_obs[env_id]['observation'] for _ in range(self.policy_config.model.frame_stack_num)], maxlen=self.policy_config.model.frame_stack_num)
                        game_segments[env_id].reset(observation_window_stack[env_id])
                        last_game_segments[env_id] = None
                        last_game_priorities[env_id] = None

                    # ログ用統計の更新
                    self_play_moves_max = max(self_play_moves_max, eps_steps_lst[env_id])
                    if not collect_with_pure_policy:
                        self_play_visit_entropy.append(visit_entropies_lst[env_id] / eps_steps_lst[env_id])
                    self_play_moves += eps_steps_lst[env_id]
                    self_play_episodes += 1

                    pred_values_lst[env_id] = []
                    search_values_lst[env_id] = []
                    eps_steps_lst[env_id] = 0
                    visit_entropies_lst[env_id] = 0

                    # Env reset は env_manager が処理するため、こちらではポリシー等をリセット
                    self._policy.reset([env_id])
                    self._reset_stat(env_id)
                    ready_env_id.remove(env_id)

            if collected_episode >= n_episode:
                # [data, meta_data]
                return_data = [self.game_segment_pool[i][0] for i in range(len(self.game_segment_pool))], [
                    {
                        'priorities': self.game_segment_pool[i][1],
                        'done': self.game_segment_pool[i][2],
                        'unroll_plus_td_steps': self.unroll_plus_td_steps
                    } for i in range(len(self.game_segment_pool))
                ]
                self.game_segment_pool.clear()
                break

        collected_duration = sum([d['time'] for d in self._episode_info])

        # 分散学習環境（DDP）が有効な場合は allreduce で集計
        if self._world_size > 1:
            self._logger.info(f"Rank {self._rank} before allreduce: collected_step={collected_step}, collected_episode={collected_episode}")
            collected_step = allreduce_data(collected_step, 'sum')
            collected_episode = allreduce_data(collected_episode, 'sum')
            collected_duration = allreduce_data(collected_duration, 'sum')
            self._logger.info(f"Rank {self._rank} after allreduce: collected_step={collected_step}, collected_episode={collected_episode}")

        self._total_envstep_count += collected_step
        self._total_episode_count += collected_episode
        self._total_duration += collected_duration

        # log
        self._output_log(train_iter)
        return return_data

    def _output_log(self, train_iter: int) -> None:
        """
        Overview:
            Log the collector's data and output the log information.
        Arguments:
            - train_iter (:obj:`int`): Current training iteration number for logging context.
        """
        if self._rank != 0:
            return
        if (train_iter - self._last_train_iter) >= self._collect_print_freq and len(self._episode_info) > 0:
            self._last_train_iter = train_iter
            episode_count = len(self._episode_info)
            envstep_count = sum([d['step'] for d in self._episode_info])
            duration = sum([d['time'] for d in self._episode_info])
            episode_reward = [d['reward'] for d in self._episode_info]
            if "world_model_cfg" in self.policy_config.model and self.policy_config.model.world_model_cfg.obs_type == 'text':
                episode_bleu = [d['text_bleu'] for d in self._episode_info]

            if not self.collect_with_pure_policy:
                visit_entropy = [d['visit_entropy'] for d in self._episode_info]
            else:
                visit_entropy = [0.0]
            if self.policy_config.gumbel_algo:
                completed_value = [d['completed_value'] for d in self._episode_info]
            self._total_duration += duration
            info = {
                'episode_count': episode_count,
                'envstep_count': envstep_count,
                'avg_envstep_per_episode': envstep_count / episode_count,
                'avg_envstep_per_sec': envstep_count / duration,
                'avg_episode_per_sec': episode_count / duration,
                'collect_time': duration,
                'reward_mean': np.mean(episode_reward),
                'reward_std': np.std(episode_reward),
                'reward_max': np.max(episode_reward),
                'reward_min': np.min(episode_reward),
                'total_envstep_count': self._total_envstep_count,
                'total_episode_count': self._total_episode_count,
                'total_duration': self._total_duration,
                'visit_entropy': np.mean(visit_entropy),
            }
            if "world_model_cfg" in self.policy_config.model and self.policy_config.model.world_model_cfg.obs_type == 'text':
                info.update({'text_avg_bleu':np.mean(episode_bleu)})
            if self.policy_config.gumbel_algo:
                info['completed_value'] = np.mean(completed_value)
            self._episode_info.clear()
            self._logger.info("collect end:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()])))
            
            for k, v in info.items():
                if k in ['each_reward']:
                    continue
                self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
                if k in ['total_envstep_count']:
                    continue
                self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, self._total_envstep_count)

            if self.policy_config.use_wandb:
                wandb.log({'{}_step/'.format(self._instance_name) + k: v for k, v in info.items()}, step=self._total_envstep_count)
