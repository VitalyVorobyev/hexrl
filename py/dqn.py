""" Deep Q-learning agent
    Inspired by
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from hexgame import HexDriver

Transition = namedtuple('Transition',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory:
    """ A cyclic buffer that holds the transitions observed recently """
    def __init__(self, capacity:int, seed:int=0) -> None:
        self.memory = deque([], maxlen=capacity)
        self.rng = np.random.default_rng(seed=seed)
    
    def push(self, *args):
        """ Save a transition """
        self.memory.append(Transition(*args))

    def sample(self, batch_size:int):
        return self.rng.choice(self.memory, size=batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """ Model that estimates sum of future rewards for a given position """
    def __init__(self, n_observations) -> None:
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return F.sigmoid(self.layer3(x))

class Trainer:
    """ Reward model trainer """
    def __init__(self, board_size:int, device:torch.device, cache_size:int, seed:int=137) -> None:
        self.rng = np.random.default_rng(seed=seed)
        self.device = device
        self.memory = ReplayMemory(cache_size, self.rng)
    
        self.batch_size = 128
        self.gamma = 0.99
        self.epshi = 0.90
        self.epslo = 0.05
        self.decay = 1000
        self.tau = 0.005
        self.lr = 1e-4

        self.board_size = board_size
        self.n_actions = board_size**2
        self.n_observations = board_size**2

        policy_net = DQN(self.n_observations)
        target_net = DQN(self.n_observations)
        target_net.load_state_dict(policy_net.state_dict())

        self.optimizer = optim.AdamW(policy_net.parameters(), lr=self.lr, amsgrad=True)

        self.steps_done = 0
        self.episode_duration = []

    def eps_threshold(self) -> float:
        """ Random action probability """
        return self.epslo + (self.epshi - self.epslo) * np.exp(-self.steps_done / self.decay)

    def select_action(self, board:HexDriver):
        """ Action policy """
        self.steps_done += 1
        if self.rng.random() > self.eps_threshold():
            with torch.no_grad():
                return self.policy_net(board.position.ravel()).max(1)[1].view(1, 1)
        else:
            q, r = board.make_random_move()
            hsize = self.board_size // 2
            action_index = (q + hsize) * self.board_size + r + hsize
            return torch.tensor([[action_index]], device=self.device, dtype=torch.long)


def main():
    pass
