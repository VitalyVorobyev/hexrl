""" Reinforcement learning enviroment based on Hex game """

import numpy as np
from hexgame import HexDriver

class HexEnv:
    """ Reinforcement learning enviroment based on Hex game """
    def __init__(self, size=int) -> None:
        self.board = HexDriver(size)
        self.size, self.hsize = size, size // 2
    
    def reset(self) -> np.ndarray:
        """ Back to initial position """
        self.board = HexDriver(self.board.position.shape[0])
        return self.board.position.ravel()

    def get_respose(self) -> tuple[np.ndarray, int, bool]:
        """ Current position, reward, and end game flag """
        if self.board.is_game_over():
            return self.board.position.ravel(), 1, True
        self.board.make_random_move()
        reward, is_endgame = ((-1, True) if self.board.is_game_over() else (0, False))
        return self.board.position.ravel(), reward, is_endgame

    def step(self, action:int) -> tuple[np.ndarray, int, bool]:
        """ Make move """
        r = (action // self.size) - self.hsize
        q = (action %  self.size) - self.hsize
        valid_move = self.board.make_move(q, r)
        assert valid_move
        return self.get_respose()

    def sample(self) -> int:
        """ Suggest random action """
        r, q = self.board.suggest_random_move()
        return (q + self.hsize) * self.size + r + self.hsize
    
    def available_actions(self) -> np.ndarray:
        """ Indices of available actions """
        return np.flatnonzero(self.board.available_moves_mask().ravel())

def main():
    """ Test program """
    size = 3
    henv = HexEnv(size)
    posi = henv.board.next_positions()
    print(posi.shape)

    henv.board.make_random_move()
    henv.board.make_random_move()
    henv.board.make_random_move()
    print(henv.available_actions())

    # for pos in posi:
    #     print(HexDriver.position_as_str(pos.reshape(-1, size)))

if __name__ == '__main__':
    main()
