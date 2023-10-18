""" Demo for gymnasium cart pole """

import sys
from itertools import count

import gymnasium as gym
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

from rltutorial import DQN

def select_action(model, state):
    """ Action policy """
    with torch.no_grad():
        return model(state).max(1)[1].view(1, 1)

def run(env, model, eposodes):
    """ Run demo """
    for epi in range(eposodes):
        print(f'Episode {epi:4d}/{eposodes}...', end=' ')
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(model, state)
            observation, _, terminated, truncated, _ = env.step(action.item())
            if terminated or truncated:
                print(f'time: {t}')
                break
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

def main():
    """ Test program """
    level = 800 if len(sys.argv) < 2 else int(sys.argv[1])

    env = gym.make('CartPole-v1', render_mode="human")
    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    state = torch.load(f'./data/CartPole_model{level}')
    model = DQN(n_observations, n_actions).to(device)
    model.load_state_dict(state)

    run(env, model, 100)


if __name__ == '__main__':
    main()
