import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt
import quantstats as qs
import pandas as pd

#  Hyperparameters
window_size = 20
start_index = window_size
day_begin = 2000
day_end = 2200
env = gym.make('stocks-v0',
               df = STOCKS_GOOGL,
               window_size = window_size,
               frame_bound = (day_begin, day_end))
observation = env.reset()

while True:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()
    print("info:", info)
    if done:
        print("info:", info)
        break

plt.cla()
env.render_all()
plt.show()


qs.extend_pandas()

net_worth = pd.Series(env.history['total_profit'], index=STOCKS_GOOGL.index[day_begin+1:day_end])
returns = net_worth.pct_change().iloc[1:]

qs.reports.full(returns)
qs.reports.html(returns, output='random_quantstats.html')
