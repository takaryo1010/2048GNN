
import torch
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from lzero.model.stochastic_muzero_model import StochasticMuZeroModel
from zoo.game_2048.envs.game_2048_env import Game2048Env

# 学習済みモデルのパス
CKPT_PATH = 'data_stochastic_mz/game_2048_npct-2_stochastic_muzero_ns100_upc200_rer0.0_bs512_chance-True_sslw2_seed0_250724_052507/ckpt/ckpt_best.pth.tar'

# 2048環境の作成
cfg = Game2048Env.default_config()
cfg.render_mode = 'image_realtime_mode'
cfg.env_id = 'game_2048'
env = Game2048Env(cfg)

model = StochasticMuZeroModel(
    observation_shape=(16, 4, 4),
    action_space_size=4,
    chance_space_size=32,
    num_res_blocks=1,
    num_channels=64,
    reward_head_hidden_channels=[32],
    value_head_hidden_channels=[32],
    policy_head_hidden_channels=[32],
    res_connection_in_dynamics=True,
    self_supervised_learning_loss=True
)
ckpt = torch.load(CKPT_PATH, map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()

obs = env.reset()
done = False
total_reward = 0

env.close()
while True:
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict) and 'observation' in obs:
        obs_tensor = torch.tensor(obs['observation'], dtype=torch.float32).unsqueeze(0)
    else:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model.initial_inference(obs_tensor)
        action = torch.argmax(output.policy_logits, dim=1).item()
    obs, reward, done, info = env.step(action)
    env.render()
    plt.pause(0.001)
    total_reward += reward
    # time.sleep(0.01)  # 表示速度調整（削除で最速）
    if done:
        env.close()
        print(f"Episode finished. Total reward: {total_reward}")
        break

env.close()
plt.show()
