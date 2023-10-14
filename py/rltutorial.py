""" This code follows torch tutorial
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import sys
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    """ Cyclic buffer of recently observed transitions """
    def __init__(self, capacity:int, rng:np.random.Generator) -> None:
        self.memory = deque([], maxlen=capacity)
        self.rng = rng

    def push(self, *args) -> None:
        """ Save a transition """
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size:int) -> np.ndarray:
        """ Get batch of random transitions from memory """
        return random.sample(self.memory, batch_size)
        # return self.rng.choice(self.memory, batch_size)
    
    def __len__(self) -> int:
        """ Number of saved states """
        return len(self.memory)

class DQN(nn.Module):
    """ Q-policy network """
    def __init__(self, n_observations:int, n_actions:int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x:np.ndarray) -> np.ndarray:
        """ Forward pass """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


class Trainer:
    """ Policy trainer """
    def __init__(self, env:gym.Env, device:torch.device, cache_size:int, seed:int) -> None:
        self.rng = np.random.default_rng(seed=seed)
        self.env = env
        self.device = device
        self.memory = ReplayMemory(cache_size, self.rng)

        self.batch_size = 128
        self.gamma = 0.99
        self.epshi = 0.90
        self.epslo = 0.05
        self.decay = 1000
        self.tau = 0.005
        self.lr = 1e-4

        n_actions = env.action_space.n
        state, info = env.reset()
        n_observations = len(state)

        print(f'{n_actions} actions')
        print(f'{n_observations} observations')
        print(f'Info:\n{info}')

        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
        self.steps_done = 0
        self.episode_durations = []

    def eps_threshold(self) -> float:
        """ Random action probability """
        return self.epslo + (self.epshi - self.epslo) *\
            np.exp(-self.steps_done / self.decay)

    def select_action(self, state):
        """ Action policy """
        self.steps_done += 1
        if self.rng.random() > self.eps_threshold():
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]],
                                device=self.device, dtype=torch.long)

    def optimize(self, optimizer):
        """ Run train step """
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Ortimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        optimizer.step()

    def run(self, num_episodes:int):
        """ Run training loop """
        optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        for i_episode in range(num_episodes):
            print(f'episode {i_episode:5d}/{num_episodes}')
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                next_state = None if terminated else\
                    torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)
                state = next_state

                self.optimize(optimizer)

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau +\
                        target_net_state_dict[key] * (1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break
                
        print('Complete')
        self.plot_durations(True)
        # plt.ioff()

    def plot_durations(self, show_result:bool=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        plt.pause(0.001)

def main():
    """ Test program """
    env = gym.make('CartPole-v1', render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(env, device, 10000, 137)

    nruns = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    trainer.run(nruns)

    model = trainer.target_net

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    torch.save(model.state_dict(), f'./data/CartPole_model{nruns}')
    plt.show()

if __name__ == '__main__':
    main()
