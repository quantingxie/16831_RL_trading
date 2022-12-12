import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import sys

sys.path.append('rob831/envs/custom_trading')

from render.stock_trading_graph import StockTradingGraph

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000
INITIAL_ACCOUNT_BALANCE = 10000

LOOKBACK_WINDOW_SIZE = 40

def factor_pairs(val):
    return [(i, val / i) for i in range(1, int(val**0.5)+1) if val % i == 0]



class StockTradingEnv(gym.Env):

  """A stock trading environment for OpenAI gym"""
  metadata = {'render.modes': ['live', 'file', 'none']}
  visualization = None

  def __init__(self, df):
    

    super(StockTradingEnv, self).__init__()

    self.df = self._adjust_prices(df)
    self.reward_range = (0, MAX_ACCOUNT_BALANCE) 

    # Actions of the format Buy x%, Sell x%, Hold, etc.
    self.action_space = spaces.Box(
      low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float32)

    # self.action_space = spaces.Discrete(
    #     2)
    # Prices contains the OHCL values for the last five prices
    # self.observation_space = spaces.Box(
    #   low=0, high=1, shape=(6, 6), dtype=np.float16)
    self.observation_space = spaces.Box(
            low=0, high=1, shape=(5 * LOOKBACK_WINDOW_SIZE + 10, ), dtype=np.float32)


  def _adjust_prices(self, df):

        adjust_ratio = df['Adjusted_Close'] / df['Close']

        df['Open'] = df['Open'] * adjust_ratio
        df['High'] = df['High'] * adjust_ratio
        df['Low'] = df['Low'] * adjust_ratio
        df['Close'] = df['Close'] * adjust_ratio

        return df


    # print ("xxxxxxxxxxxxxx", self.action_space.shape)
    # print ("yyyyyyyyyyyyyy", self.observation_space.shape)
  def _next_observation(self):
        # # Get the stock data points for the last 5 days and scale to between 0-1
        # frame = np.array([
        #     self.df.loc[self.current_step: self.current_step +
        #                 5, 'Open'].values / MAX_SHARE_PRICE,
        #     self.df.loc[self.current_step: self.current_step +
        #                 5, 'High'].values / MAX_SHARE_PRICE,
        #     self.df.loc[self.current_step: self.current_step +
        #                 5, 'Low'].values / MAX_SHARE_PRICE,
        #     self.df.loc[self.current_step: self.current_step +
        #                 5, 'Close'].values / MAX_SHARE_PRICE,
        #     self.df.loc[self.current_step: self.current_step +
        #                 5, 'Volume'].values / MAX_NUM_SHARES,
        # ])

        # # Append additional data and scale each value to between 0-1
        # obs = np.append(frame, [[
        #     self.balance / MAX_ACCOUNT_BALANCE,
        #     self.max_net_worth / MAX_ACCOUNT_BALANCE,
        #     self.shares_held / MAX_NUM_SHARES,
        #     self.cost_basis / MAX_SHARE_PRICE,
        #     self.total_shares_sold / MAX_NUM_SHARES,
        #     self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        # ]], axis=0)

        frame = np.zeros((5, LOOKBACK_WINDOW_SIZE + 1))

        # Get the stock data points for the last 5 days and scale to between 0-1
        np.put(frame, [0, 4], [
            self.df.loc[self.current_step: self.current_step +
                        LOOKBACK_WINDOW_SIZE, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        LOOKBACK_WINDOW_SIZE, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        LOOKBACK_WINDOW_SIZE, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        LOOKBACK_WINDOW_SIZE, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        LOOKBACK_WINDOW_SIZE, 'Volume'].values / MAX_NUM_SHARES,
        ])

        # print("frame ==", frame.shape)

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [
            [self.balance / MAX_ACCOUNT_BALANCE],
            [self.max_net_worth / MAX_ACCOUNT_BALANCE],
            [self.shares_held / MAX_NUM_SHARES],
            [self.cost_basis / MAX_SHARE_PRICE],
            [self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE)],
        ], axis=1)



        stacked_obs = obs.flatten(order='C')
        print("stacked obs ==", stacked_obs)
        return stacked_obs

  def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        action_type = action[0]
        amount = action[1]

        # action_type = action
        # amount = 1

        # print("action taken", action)
        # print("amount each share", amount)


        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost

            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

            # Whenever we buy or sell shares, we are now going to add the details 
            # of that transaction to the self.trades object, which we’ve 
            # initialized to [] in our reset method.
            if shares_bought > 0:
                self.trades.append({'step': self.current_step,
                    'shares': shares_bought, 'total': additional_cost,
                    'type': "buy"})


        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

            # Append sell information to self.trade object
            if shares_sold > 0:
                self.trades.append({'step': self.current_step,
                    'shares': shares_sold, 'total': shares_sold * current_price,
                    'type': "sell"})



        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

        #return self.balance, self.max_net_worth, self.shares_held ,self.cost_basis, self.total_shares_sold, self.total_sales_value

  def step(self, action):
        # Execute one time step within the environment

        # print("Actionxxxxx", action)
        self._take_action(action)

        # self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        obs = self._next_observation()

        reward, done = self.get_reward(obs, action)

        assert obs.shape == (5 * LOOKBACK_WINDOW_SIZE + 10,), 'obseervation form step is not 1 dimensional'

        return obs, reward, done, {}


# Called any time a new environment is created or to reset an existing 
# environment’s state. It’s here where we’ll set the starting balance of 
# each agent and initialize its open positions to an empty list.
  def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Open'].values) - 6)

        self.trades = []
        n_obs = self._next_observation()

        assert n_obs.shape == (5 * LOOKBACK_WINDOW_SIZE + 10,), 'observation form reset() is not 1 dimensional'

        return n_obs

# Simple  render method just to print the profit
#   def render(self, mode='human', close=False):
#         # Render the environment to the screen
#         profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

#         print(f'Step: {self.current_step}')
#         print(f'Balance: {self.balance}')
#         print(
#             f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
#         print(
#             f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
#         print(
#             f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
#         print(f'Profit: {profit}')

  def _render_to_file(self, filename='render.txt'):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
  
        file = open(filename, 'a+')
        file.write(f'Step: {self.current_step}\n')
        file.write(f'Balance: {self.balance}\n')
        file.write(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})\n')
        file.write(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})\n')
        file.write(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})\n')
        file.write(f'Profit: {profit}\n\n')
        file.close()
    
# Fancy render method, takes in mode
  def render(self, mode='live', title=None, **kwargs):
  # Render the environment to the screen
        if mode == 'file':
            # What are the meaning of kwargs.get? 
            self._render_to_file(kwargs.get('filename', 'render.txt'))

        elif mode == 'live':
            if self.visualization == None:
                self.visualization = StockTradingGraph(self.df, title)
    
            if self.current_step > LOOKBACK_WINDOW_SIZE:        
                self.visualization.render(self.current_step, self.net_worth, 
                self.trades, window_size=LOOKBACK_WINDOW_SIZE)

# Create a calculate_reward function that need to take observations and actions as input
  # input a sequence of actions and predicted observations
  def get_reward(self, observations, actions):
    if(len(observations.shape)==1):
        observations = np.expand_dims(observations, axis = 0)
        actions = np.expand_dims(actions, axis = 0)
        batch_mode = False
    else:
        batch_mode = True
    
    # print("observations", observations, observations.shape)

    balance = observations[:,5 * LOOKBACK_WINDOW_SIZE + 5]
    net_worth = observations[:, 5 * LOOKBACK_WINDOW_SIZE + 6]

    print("net_worth in get_reward", balance)
    self.current_step += 1

    delay_modifier = (self.current_step / MAX_STEPS)

    self.reward = balance * delay_modifier + self.current_step
    # Sum up all the balance and normalize it
    #self.reward = np.sum(balance)/len(balance)

    

    # print("balance", balance, balance.shape)


    #delay_modifier = (self.current_step / MAX_STEPS)
    
    #done
    dones = net_worth <= 0


    if(not batch_mode):
        return self.reward[0], dones[0]
    return self.reward, dones
   
   
  def close(self):
    if self.visualization != None:
        self.visualization.close()
        self.visualization = None