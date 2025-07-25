import gym
import torch
import time
from lzero.model.muzero_model_mlp import MuZeroModelMLP

# 学習済みモデルのパス
CKPT_PATH = 'LightZero/iteration_6900.pth.tar'

# CartPole環境の作成
env = gym.make('CartPole-v1', render_mode='human')

model = MuZeroModelMLP(
    observation_shape=4,
    action_space_size=2,
    latent_state_dim=128,
    reward_head_hidden_channels=[32],
    value_head_hidden_channels=[32],
    policy_head_hidden_channels=[32],
    common_layer_num=2,
    res_connection_in_dynamics=True,
    self_supervised_learning_loss=True
)
ckpt = torch.load(CKPT_PATH, map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()

obs = env.reset()
done = False
total_reward = 0

while True:
    if isinstance(obs, tuple):
        obs = obs[0]  # gymnasium対応
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model.initial_inference(obs_tensor)
        action = torch.argmax(output.policy_logits, dim=1).item()
    obs, reward, done, info = env.step(action)
    env.render()
    total_reward += reward
    time.sleep(0.05)  # 表示速度調整
    if done:
        print(f"Episode finished. Total reward: {total_reward}")
        break

env.close()