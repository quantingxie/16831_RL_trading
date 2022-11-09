import gym
import gym_anytrading

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions

# Processing libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import quantstats as qs


window_size = 30
start_index = window_size
end_train_date = 1500

# df = pd.read_csv('/home/quanting/PycharmProjects/16831_RL_trading/Reinforcement-Learning-for-Trading-main/data/gmedata.csv')
#
# df['Date'] = pd.to_datetime(df['Date'])
#
# df.set_index('Date', inplace=True)
# Making the environment using data from gym_anytrading

# def my_process_data(env):
#     start = env.frame_bound[0] - env.window_size
#     end = env.frame_bound[1]
#     prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
#     signal_features = env.df.loc[:, ['Close', 'Open', 'High', 'Low']].to_numpy()[start:end]
#     return prices, signal_features
#
#
# class MyForexEnv(StocksEnv):
#     _process_data = my_process_data


env = gym.make('stocks-v0',
                df=STOCKS_GOOGL,
                window_size=window_size,
                frame_bound=(window_size, end_train_date))

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)
model.save('models/PPO_model')
