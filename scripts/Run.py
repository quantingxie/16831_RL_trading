import pandas as pd
import numpy as np
import gym_anytrading
import gym
from matplotlib import pyplot as plt
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL

import quantstats as qs
from PPO import model


window_size = 20
start_index = window_size
end_train_date = 2000
end_test_date = 2200

model.load('models/PPO_model')
env = gym.make('stocks-v0', df=STOCKS_GOOGL, frame_bound=(end_train_date,end_test_date), window_size=window_size)
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
# env.render_all()
plt.show()

qs.extend_pandas()

net_worth = pd.Series(env.history['total_profit'], index=STOCKS_GOOGL.index[end_train_date+1:end_test_date])
returns = net_worth.pct_change().iloc[1:]

pd.DataFrame(env.history['total_profit']).to_csv("profit_PPO.csv")

print(returns)
qs.reports.full(returns)
qs.reports.html(returns, output='PPO_quantstats.html')