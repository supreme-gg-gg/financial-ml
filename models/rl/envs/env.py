import gymnasium as gym, numpy as np
from gymnasium import spaces
import pandas as pd
from models.rl.utils import get_data

'''
The environment will not return single feature vectors for each observation.
Instead, it keeps a rolling window / buffer of the last N observations
so that it can reutrn a sequence of observations of length N.
This makes the training process simpler.
'''

# TO BE WRITTEN: Importing data and processing it so the env can use it

SEQUENCE_LENGTH = 20
TRADING_COST_BPS = 1e-3

# Modify and see how to use this for reward function instead!!
def _sharpe(Returns, freq=252):
    """Given a set of returns, calculates naive (rfr=0) sharpe """
    return (np.sqsrt(freq)*np.mean(Returns)) / np.std(Returns)

class TradingEnv(gym.Env):

    '''
    - bod_position = begin of day position -- Short (0), Flat(1), Long(2)
    - positions (array) = a list that stores the position of agent over time
    '''

    def __init__(self, steps, sequence_length=SEQUENCE_LENGTH):
        super(TradingEnv, self).__init__()
        self.sequence_length = sequence_length
        self.steps = steps
        self.current_step = 0
        self.asset = 10_000
        self.rtn = np.ones(self.steps)
        self.benchmark_rtn = 0 # Buy and hold benchmark return
        self.actions = np.zeros(self.steps)
        self.mkt_rtn = np.zeros(self.steps)
        self.data = get_data()
        self.trades = np.zeros(self.steps)
        self.positions = np.zeros(self.steps)


        # Action and observation space
        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(sequence_length, 10), dtype=np.float32)

        # Initialize state buffer
        self.state_buffer = np.zeros((sequence_length, 10))

    def reset(self):
        
        # TODO: We need to come back to initialize stuff properly!

        self.current_step = 0
        self.asset = 10_000
        self.rtn = 0
        self.benchmark_rtn = 0
        # Reset the state buffer
        self.state_buffer = np.zeros((self.sequence_length, 10))
        return self.state_buffer

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))   

        # Update the current step
        obs = self.data.iloc[self.current_step]
        done = self.check_completion()
        self.state_buffer = np.roll(self.state_buffer, shift=-1, axis=0)
        self.state_buffer[-1] = obs

        # This represents the position at the beginning of the day
        bod_position = 0.0 if self.current_step ==0 else self.positions[self.current_step -1]
        
        reward = 0  # Compute the reward based on the action

        self.mkt_rtn[self.current_step] = obs['Return']
        self.actions[self.current_step] = action

        self.positions[self.current_step] = action - 1
        self.trades[self.current_step] = self.positions[self.current_step] - bod_position
        self.costs[self.current_step] = abs(self.trades[self.current_step]) * TRADING_COST_BPS
        # For now we don't use time cost to simplify the problem
        reward = ((bod_position * obs['Return']) - self.costs[self.current_step])
        self.rtn[self.current_step] = reward

        self.current_step += 1

        return self.state_buffer, reward, done, {}
    
    def check_completion(self):
        if self.current_step >= self.steps:
            return True
        elif self.asset <= 0:
            return True
        elif self.rtn >= self.benchmark_rtn * 2:
            return True
        else:
            return False

    def calculate_rewards(self):
        pass

    def calculate_returns(self):
        pass

# Example usage
env = TradingEnv()
state = env.reset()
action = env.action_space.sample()
next_state, reward, done, _ = env.step(action)
