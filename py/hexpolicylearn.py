"""  """

import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from hexenv import HexEnv

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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

    def __len__(self) -> int:
        """ Number of saved states """
        return len(self.memory)

class HexPolicyNet(nn.Module):
    """ Q-policy network for the Hex game """
    def __init__(self, board_size:int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(board_size**2, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, board_size**2)
        self.layers = [
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        ]

    def forward(self, x:np.ndarray) -> np.ndarray:
        """ Forward pass """
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

class Trainer:
    """ Policy trainer """
    def __init__(self, env:HexEnv, device:torch.device, cache_size:int, seed:int) -> None:
        self.rng = np.random.default_rng(seed=seed)
        self.env = env
        self.device = device
        self.memory = ReplayMemory(cache_size, self.rng)

        self.fig, self.ax, self.line, self.hist = None, None, None, None

        self.batch_size = 128
        self.gamma = 0.99
        self.epshi = 0.90
        self.epslo = 0.05
        self.decay = 1000
        self.tau = 0.005
        self.lr = 1e-4

        self.policy_net = HexPolicyNet(env.size).to(device)
        self.target_net = HexPolicyNet(env.size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done = 0
        self.game_outcome = []

    def init_train_plot(self, num_episodes:int) -> None:
        """ Create objects for train plot """
        self.fig, self.ax = plt.subplots(ncols=2, figsize=(11, 5))
        self.ax[0].minorticks_on()
        self.ax[0].grid(which='minor', linestyle=':')
        self.ax[0].grid(which='major')
        self.ax[0].set_xlabel('Episode')
        self.ax[0].set_ylabel('Game outcome')
        self.ax[0].set_ylim((-1.2, 1.2))

        self.ax[1].minorticks_on()
        self.ax[1].grid(which='minor', linestyle=':')
        self.ax[1].grid(which='major')
        self.ax[1].set_ylabel('Victory rate')
        self.ax[1].set_ylim((0, 1.05))
        self.ax[1].set_xlim((-0.1, 0.1))

        x = np.zeros(num_episodes)
        self.line, = self.ax[0].plot(x, 'o', markersize=1)
        self.hist, = self.ax[1].plot([0.5], 'o', markersize=7)
        self.fig.tight_layout()
    
    def update_train_plot(self, num_episodes:int):
        """ Update traning plot """
        newy = np.zeros(num_episodes)
        newy[:len(self.game_outcome)] = self.game_outcome
        self.line.set_ydata(newy)
        self.hist.set_ydata([np.mean(np.array(self.game_outcome[-100:]) > 0)])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def eps_threshold(self) -> float:
        """ Random action probability """
        return self.epslo + (self.epshi - self.epslo) *\
            np.exp(-self.steps_done / self.decay)

    def select_action(self, state) -> int:
        """ Action policy """
        self.steps_done += 1
        if self.rng.random() < self.eps_threshold():
            return torch.tensor([[self.env.sample()]], device=self.device, dtype=torch.long)

        available_actions = self.env.available_actions()
        with torch.no_grad():
            r_actions = self.policy_net(state)[0, :]
            action_idx = np.argmax([r_actions[action] for action in available_actions])
            action = available_actions[action_idx]
            return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def optimize(self, optimizer):
        """ Run train step """
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)  # In-place gradient clipping
        optimizer.step()

    def update_target_state(self) -> None:
        """ Use policy_net to update target_net """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau +\
                target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def run(self, num_episodes:int):
        """ Run training loop """
        self.init_train_plot(num_episodes)
        optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        for i_episode in range(num_episodes):
            print(f'episode {i_episode:5d}/{num_episodes}')
            if i_episode % 200 == 0:
                torch.save(self.target_net.state_dict(), f'./data/Hex_model{i_episode}')
            state = torch.tensor(self.env.reset().astype(np.float32),
                                 dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, is_gameover = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)

                next_state = None if is_gameover else\
                    torch.tensor(observation.astype(np.float32),
                                 dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize(optimizer)
                self.update_target_state()

                if is_gameover:
                    self.game_outcome.append(reward.item())
                    self.update_train_plot(num_episodes)
                    break

        print('Complete')
        plt.ioff()

def main():
    """ Test program """
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    nruns = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    print(f'Run {nruns} games of board size {size}x{size}')

    plt.ion()
    assert plt.isinteractive()

    env = HexEnv(size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(env, device, 10000, 137)

    trainer.run(nruns)

    model = trainer.target_net
    torch.save(model.state_dict(), f'./data/CartPole_model{nruns}')
    plt.show()

if __name__ == '__main__':
    main()
