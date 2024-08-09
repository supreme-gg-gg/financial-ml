import random, math, json, gym
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import plot_durations

# ======================Load Hyperparameters=======================

with open("config.json", "r") as f:
    config = json.load(f)

MEMORY_SIZE = config["MEMORY_SIZE"]
BATCH_SIZE = config["BATCH_SIZE"]
GAMMA = config["GAMMA"]
EPS = config["EPS"]
EPS_MIN = config["EPS_MIN"]
EPS_DECAY = config["EPS_DECAY"]
TAU = config["TAU"]
LR = config["LR"]

# ======================Replay Buffer=======================

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
# ======================Neural Network=======================

# For now we assume transitions are totally deterministic

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# ======================Agent=======================

class DQNAgent():

    def __init__(self, env, device):
        n_observations = len(env.reset()[0])
        n_actions = env.action_space.n

        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        self.env = env
        self.device = device
        self.n_actions = n_actions
        self.n_observations = n_observations
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0
    
    def select_action(self, state):

        eps_threshold = EPS_MIN + (EPS - EPS_MIN) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1,1)
        else:
            action = self.env.action_space.sample()
            return torch.tensor([[action]], device=self.device, dtype=torch.long)
    
    def optimize_model(self):

        if len(self.memory) < BATCH_SIZE:
            return
        
        # Sample a random batch of transitions and unpack them into their own tensors
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Also track non-final states and their indices (basically the "done" field)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s') for all next states, expected values of actions for 
        # non_final_next_states are based on the older target_net, select best reward
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Unlike the TF example we are using Huber Loss here (look it up)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train_agent(self, num_episodes):

        '''
        Args:
        - num_episodes (int): number of episodes to train the agent

        Returns:
        - episode_durations (list): list containing the duration of each episode
        '''

        episode_durations = []

        for _ in range(num_episodes):

            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model() # on the policy network

                # soft update target network weights
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in target_net_state_dict:
                    target_net_state_dict[key] = TAU * policy_net_state_dict[key] + (1 - TAU) * target_net_state_dict[key]
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    # duration is how long it took to run that episode
                    episode_durations.append(t + 1)
                    plot_durations(episode_durations)
                    break
    
        return episode_durations

print("DQN Module Loaded")