""" Reinforcement learning enviroment based on Hex game """

import numpy as np
from hexgame import HexDriver

class HexEnv:
    """ Reinforcement learning enviroment based on Hex game """
    def __init__(self, size=int) -> None:
        self.board = HexDriver(size)
    
    def reset(self):
        """ Back to initial position """
        self.board = HexDriver(self.board.position.shape[0])

    def get_respose(self) -> tuple[np.ndarray, int, bool]:
        if self.board.is_game_over():
            return self.board.position, 1, True
        self.board.make_random_move()
        reward, endgame = -1, True if self.board.is_game_over() else 0, False
        return self.board.position, reward, endgame

    def step(self, move:int) -> tuple[np.ndarray, int, bool]:
        """ Make move """
        q = self.board.position.size // move
        r = self.board.position.size %  move
        valid_move = self.board.make_move(q, r)
        assert valid_move
        return self.get_respose()

    def legal_moves_and_corresponding_positions(self) -> list:
        pass

    def sample(self) -> tuple[np.ndarray, int, bool]:
        """ Make random move """
        self.board.make_random_move()
        return self.get_respose()


