import logging
logger = logging.getLogger(__name__)
import os
from functools import partial
from typing import Optional, Tuple
from typing import TYPE_CHECKING

import torch
import wandb
from ding.config import compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

# lzero 内部ユーティリティ関数を読み込み
from lzero.entry.utils import log_buffer_memory_usage, log_buffer_run_time
from lzero.policy import visit_count_temperature
from lzero.policy.random_policy import LightZeroRandomPolicy
# MuZero 用の Collector / Evaluator をエイリアスで読み込む
from lzero.worker import MuZeroCollector as Collector
from lzero.worker import MuZeroEvaluator as Evaluator
from .utils import random_collect, calculate_update_per_collect

if TYPE_CHECKING:
    # 型チェック時のみ Policy 型をインポート（実行時の未定義エラー回避）
    from ding.policy import Policy  # type: ignore


def train_muzero(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    MuZero / EfficientZero 系アルゴリズムの学習エントリポイント

    引数:
        input_cfg: Tuple[dict, dict]
            - (user_config, create_cfg) のタプル。ユーザ設定と作成用設定が入る。
        seed: ランダムシード（デフォルト 0）
        model: 事前構築済みの torch.nn.Module（省略可）
        model_path: 読み込む事前学習済みモデルのパス（省略可）。通常は "exp_name/ckpt/ckpt_best.pth.tar" のようなファイル。
        max_train_iter: 学習更新の最大反復数（デフォルトは非常に大きな数）
        max_env_step: 環境との最大ステップ数（デフォルトは非常に大きな数）

    戻り値:
        policy: 学習後の Policy オブジェクト
    """

    # input_cfg は (cfg, create_cfg) の形を期待する
    cfg, create_cfg = input_cfg

    # create_cfg.policy.type がサポート対象かをチェック
    assert create_cfg.policy.type in [
        'efficientzero', 'muzero', 'muzero_context', 'muzero_rnn_full_obs',
        'sampled_efficientzero', 'sampled_muzero', 'gumbel_muzero', 'stochastic_muzero'
    ], \
        "train_muzero エントリは現在以下のアルゴリズムのみサポートします: 'efficientzero', 'muzero', 'sampled_efficientzero', 'gumbel_muzero', 'stochastic_muzero'"

    # create_cfg.policy.type に応じて、適切な GameBuffer クラスを遅延インポートして選択する
    if create_cfg.policy.type in ['muzero', 'muzero_context', 'muzero_rnn_full_obs']:
        # MuZero 一般用のゲームバッファ
        from lzero.mcts import MuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'efficientzero':
        # EfficientZero 用のゲームバッファ
        from lzero.mcts import EfficientZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'sampled_efficientzero':
        # サンプリング版 EfficientZero のゲームバッファ
        from lzero.mcts import SampledEfficientZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'sampled_muzero':
        # サンプリング版 MuZero のゲームバッファ
        from lzero.mcts import SampledMuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'gumbel_muzero':
        # Gumbel MuZero 用のゲームバッファ
        from lzero.mcts import GumbelMuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'stochastic_muzero':
        # 確率的 MuZero 用のゲームバッファ
        from lzero.mcts import StochasticMuZeroGameBuffer as GameBuffer


    # CUDA が利用可能かつ cfg.policy.cuda が True の場合は device を 'cuda' に設定、そうでなければ 'cpu'
    if cfg.policy.cuda and torch.cuda.is_available():
        cfg.policy.device = 'cuda'
    else:
        cfg.policy.device = 'cpu'

    # 設定のコンパイル（自動補完や保存など）。seed を設定して config を最終化する
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

    # --- 環境とポリシーの生成 ---
    # get_vec_env_setting は env 関連の工場関数と、それぞれのサブ環境設定を返す
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    # collector 用の環境マネージャを作成（各サブ設定ごとに env_fn を部分適用して渡す）
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    # evaluator 用の環境マネージャを作成
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    # 環境シードの設定（collector は動的シード有り、evaluator は固定シード）
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    # PyTorch や numpy 等のパッケージのシードを統一
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    # オフライン評価モードの場合、学習フックに checkpoint 保存頻度を設定
    if cfg.policy.eval_offline:
        cfg.policy.learn.learner.hook.save_ckpt_after_iter = cfg.policy.eval_freq

    # wandb（Weights & Biases）を使う設定が有効なら初期化
    if cfg.policy.use_wandb:
        # wandb の初期化（実験名や設定を渡す）
        wandb.init(
            project="LightZero",
            config=cfg,
            sync_tensorboard=False,
            monitor_gym=False,
            save_code=True,
        )

    # ポリシーを作成。model が与えられればそれを利用。学習、データ収集、評価の各モードを有効化
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # --- 追加ログ: モデルのクラス名とパラメータ数を訓練開始時に一度だけ出力 ---
    try:
        # policy がラッパーの場合は内部の torch モデルを持つことがあるため安全にアクセス
        policy_model = getattr(policy, "model", policy)
        model_cls_name = type(policy_model).__module__ + "." + type(policy_model).__name__
        total_params = sum(int(p.numel()) for p in policy_model.parameters())
        trainable_params = sum(int(p.numel()) for p in policy_model.parameters() if p.requires_grad)
        logger.info(
            "Model info: %s | total_params=%d | trainable_params=%d",
            model_cls_name,
            total_params,
            trainable_params,
        )
    except Exception:
        # ログ取得はデバッグ用なので失敗しても訓練を止めない
        logger.exception("Failed to fetch model info for logging")

    # 事前学習済みモデルのロード（path が与えられている場合）
    if model_path is not None:
        # map_location でデバイスを合わせてロード
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))

    # --- ワーカー（学習者・収集器・評価器・リプレイバッファ等）の作成 ---
    # TensorBoard のロガーはランク 0（主プロセス）のみ作成
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    # BaseLearner を作成。学習ループやフックを担う主要オブジェクト
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # ==============================================================
    # MCTS+RL アルゴリズムに関するコア処理
    # ==============================================================
    policy_config = cfg.policy  # 短縮変数: ポリシー設定
    batch_size = policy_config.batch_size  # ミニバッチサイズ
    # MCTS 系アルゴリズム向けのゲームバッファを初期化（上で選択した GameBuffer クラスを使用）
    replay_buffer = GameBuffer(policy_config)

    # データ収集用オブジェクト（Collector）を作成
    collector = Collector(
        env=collector_env,               # 収集用環境
        policy=policy.collect_mode,     # 収集用のポリシーモード（deterministic/parallel など）
        tb_logger=tb_logger,            # TensorBoard ロガー
        exp_name=cfg.exp_name,          # 実験名（ログ/チェックポイント用）
        policy_config=policy_config,    # ポリシーの設定
    )

    # 評価用オブジェクト（Evaluator）を作成
    evaluator = Evaluator(
        eval_freq=cfg.policy.eval_freq,                   # 評価する頻度（学習イテレーション単位）
        n_evaluator_episode=cfg.env.n_evaluator_episode, # 評価で使うエピソード数
        stop_value=cfg.env.stop_value,                    # 達成すれば学習停止する評価値
        env=evaluator_env,                                # 評価用環境
        policy=policy.eval_mode,                          # 評価用ポリシー
        tb_logger=tb_logger,                              # TensorBoard ロガー
        exp_name=cfg.exp_name,                            # 実験名
        policy_config=policy_config                       # ポリシー設定
    )

    # ==============================================================
    # メインループ
    # ==============================================================
    # 学習者の before_run フックを呼ぶ（初期化処理やログ出力など）
    learner.call_hook('before_run')

    # wandb を使っている場合は初回の学習イテレート／環境ステップを設定
    if policy_config.use_wandb:
        policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

    # 明示的に update_per_collect が設定されていればそれを使う（後で変更される可能性あり）
    if cfg.policy.update_per_collect is not None:
        update_per_collect = cfg.policy.update_per_collect

    # ランダム行動で事前に指定エピソード数だけデータ収集する目的:
    # - 探索性を高めるため
    # - ランダムポリシーのベースラインを取得するため
    if cfg.policy.random_collect_episode_num > 0:
        random_collect(cfg.policy, policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)

    # オフライン評価モードなら、評価時に使うイテレーションと envstep のリストを初期化
    if cfg.policy.eval_offline:
        eval_train_iter_list = []
        eval_train_envstep_list = []

    # ランダムエージェントの評価を一度行う（初期ベースライン）
    stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)

    # 無限ループで学習を回す（内部で終了条件をチェックして break する）
    while True:
        # バッファのメモリ使用量や実行時間をログ出力
        log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)
        log_buffer_run_time(learner.train_iter, replay_buffer, tb_logger)

        # 収集時にコントロールするキーワード引数を格納する辞書
        collect_kwargs = {}

        # visit count 分布に使う温度パラメータを、学習ステップに応じて決定
        # MuZero 論文付録の温度スケジュールを参照
        collect_kwargs['temperature'] = visit_count_temperature(
            policy_config.manual_temperature_decay,
            policy_config.fixed_temperature_value,
            policy_config.threshold_training_steps_for_final_temperature,
            trained_steps=learner.train_iter
        )

        # ε-greedy を収集時に使う設定があれば、その関数を作り現在の envstep に応じた ε を計算
        if policy_config.eps.eps_greedy_exploration_in_collect:
            epsilon_greedy_fn = get_epsilon_greedy_fn(
                start=policy_config.eps.start,
                end=policy_config.eps.end,
                decay=policy_config.eps.decay,
                type_=policy_config.eps.type
            )
            # collector の envstep に基づいて ε を取得
            collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)
        else:
            # 使用しない場合は 0（確定的選択）
            collect_kwargs['epsilon'] = 0.0

        # 評価が必要かチェックする
        if evaluator.should_eval(learner.train_iter):
            if cfg.policy.eval_offline:
                # オフライン評価モードでは、後でまとめて評価するためにイテレーションと envstep を保存
                eval_train_iter_list.append(learner.train_iter)
                eval_train_envstep_list.append(collector.envstep)
            else:
                # 通常はその場で評価を実行し、終了フラグが立てばループ脱出
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    break

        # デフォルト設定に従ってデータを収集（n_sample / n_episode ベース）
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

        # 1 回の収集あたり何回学習更新するかを計算
        update_per_collect = calculate_update_per_collect(cfg, new_data)

        # 収集したデータ（game segment）をリプレイバッファへ保存
        replay_buffer.push_game_segments(new_data)
        # バッファが満杯なら古いデータを削る
        replay_buffer.remove_oldest_data_to_fit()

        # 保存されたデータを使って学習を行うループ
        #TODO:学習について知るために、learner.train の中身を読む
        for i in range(update_per_collect):
            # 指定バッチサイズより多くの遷移があればサンプリング可能
            if replay_buffer.get_num_of_transitions() > batch_size:
                train_data = replay_buffer.sample(batch_size, policy)
            else:
                # 不足していれば警告を出して収集に戻る
                logging.warning(
                    f'The data in replay_buffer is not sufficient to sample a mini-batch: '
                    f'batch_size: {batch_size}, '
                    f'{replay_buffer} '
                    f'continue to collect now ....'
                )
                break

            # wandb を利用していれば学習イテレート／envstep を更新
            if policy_config.use_wandb:
                policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

            # 実際の学習ステップ（学習者にデータを与えて train を呼ぶ）
            log_vars = learner.train(train_data, collector.envstep)

            # 優先経験リプレイを使っている場合は、優先度を更新
            if cfg.policy.use_priority:
                # log_vars[0]['value_priority_orig'] は優先度計算に使われる生値を想定
                replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        # 終了条件: collector の環境ステップ数か学習反復数が上限を超えたら終了処理へ
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            if cfg.policy.eval_offline:
                # オフライン評価モード: 事前に保存した各イテレーションの ckpt を読み込み順次評価する
                logging.info(f'eval offline beginning...')
                ckpt_dirname = './{}/ckpt'.format(learner.exp_name)
                # 事前学習済みモデルの性能を評価するループ
                for train_iter, collector_envstep in zip(eval_train_iter_list, eval_train_envstep_list):
                    ckpt_name = 'iteration_{}.pth.tar'.format(train_iter)
                    ckpt_path = os.path.join(ckpt_dirname, ckpt_name)
                    # ckpt をロードして評価ポリシーへ反映
                    policy.learn_mode.load_state_dict(torch.load(ckpt_path, map_location=cfg.policy.device))
                    stop, reward = evaluator.eval(learner.save_checkpoint, train_iter, collector_envstep)
                    logging.info(
                        f'eval offline at train_iter: {train_iter}, collector_envstep: {collector_envstep}, reward: {reward}')
                logging.info(f'eval offline finished!')
            break

    # 学習終了後のクリーンアップフック
    learner.call_hook('after_run')
    # wandb セッションを終了
    wandb.finish()
    # 学習済みポリシーを返す
    return policy
