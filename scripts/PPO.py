import gym
import gym_anytrading

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
# Processing libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# df = pd.read_csv('/home/quanting/PycharmProjects/16831_RL_trading/Reinforcement-Learning-for-Trading-main/data/gmedata.csv')
#
# df['Date'] = pd.to_datetime(df['Date'])
#
# df.set_index('Date', inplace=True)
# Making the environment using data from gym_anytrading
env = gym.make('stocks-v0',
                df=STOCKS_GOOGL,
                window_size=10,
                frame_bound=(10, 100))

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)


env = gym.make('stocks-v0', df=STOCKS_GOOGL, frame_bound=(100,130), window_size=10)
obs = env.reset()
while True:
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        print("info", info)
        break

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()
