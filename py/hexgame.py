""" Hex game driver """

import numpy as np
import hexpex
import timeit

class HexDriver:
    """ Hex game mechanics """
    def __init__(self, size:int=11):
        """ Ctor """
        self.game_over = False
        self.moves = 0
        self.blue = 1
        self.red = -1
        self.smap = { self.blue: 'x', self.red:  'o', 0: '.' }
        self.position = np.zeros((size, size), dtype=int)
        self.rng = np.random.default_rng()

    def next_positions(self) -> np.ndarray:
        """ Generate all possible next positions """
        nmoves = self.position.size - self.moves
        rowsize = self.position.shape[0]
        positions = np.empty((nmoves, self.position.size), dtype=int)
        positions[:] = self.position.ravel()
        rows, cols = np.where(self.position == 0)
        value = self.red if self.red_moves() else self.blue
        for pos, row, col in zip(positions, rows, cols):
            pos[rowsize * row + col] = value
        return positions

    def available_moves_mask(self) -> np.ndarray:
        """ Get mask of the currently available moves """
        return self.position == 0

    def suggest_random_move(self) -> tuple[int, int]:
        """ Returns a valid move, doesn't change state of the game """
        assert not self.game_over
        if self.position.size > self.moves + 1:
            move = self.rng.integers(self.position.size - self.moves - 1, size=1).item()
        else:
            move = 0  # last move
        rows, cols = np.where(self.position == 0)
        hsize = self.position.shape[0] // 2
        q = cols[move] - hsize
        r = rows[move] - hsize
        return q, r

    def make_random_move(self) -> tuple[int, int]:
        """ Generate random move and apply it """
        q, r = self.suggest_random_move()
        self.make_move(q, r)
        return q, r

    def make_move(self, q:int, r:int) -> bool:
        """ Change game state given the next move """
        if self.game_over:
            print('No moves: game over')
            return False
        if not self.is_in_board(q, r) or self.cell_color(q, r) != 0:
            print(f'Bad move: {q:+d} {r:+d} {self.is_in_board(q, r)} {self.cell_color(q, r)}')
            return False
        hsize = self.position.shape[0] // 2
        col, row = q + hsize, r + hsize
        self.position[row, col] = self.red if self.red_moves() else self.blue

        self.moves += 1
        self.game_over = self.is_game_over()

        return True

    def is_game_over(self):
        """ True if game is finished """
        if self.game_over:
            return self.game_over
        return (self.blue_wins() if self.red_moves() else self.red_wins())

    def red_moves(self) -> bool:
        """ True if red makes the next move """
        return self.moves % 2

    def is_in_board(self, q:int|hexpex.Axial, r:int|None=None) -> bool:
        """ Checks that hex board position within the current board """
        size = self.position.shape[0]
        hsize = self.position.shape[0] // 2
        col, row = (q.q, q.r) if r is None else (q, r)
        col, row = col + hsize, row + hsize
        return col >= 0 and col < size and row >= 0 and row < size

    def cell_color(self, q:int|hexpex.Axial, r:int|None=None) -> int:
        hsize = self.position.shape[0] // 2
        col, row = (q.q, q.r) if r is None else (q, r)
        return self.position[row + hsize, col + hsize]

    def blue_wins(self) -> bool:
        """ True if a winning position for blue """
        hsize = self.position.shape[0] // 2
        sources = [hexpex.Axial(-hsize, r) for r in range(-hsize, hsize + 1) if self.cell_color(-hsize, r) == self.blue]
        targets = [hexpex.Axial( hsize, r) for r in range(-hsize, hsize + 1) if self.cell_color( hsize, r) == self.blue]
        return self.dfs(sources, targets)

    def red_wins(self) -> bool:
        """ """
        hsize = self.position.shape[0] // 2
        sources = [hexpex.Axial(q, -hsize) for q in range(-hsize, hsize + 1) if self.cell_color(q, -hsize) == self.red]
        targets = [hexpex.Axial(q,  hsize) for q in range(-hsize, hsize + 1) if self.cell_color(q,  hsize) == self.red]
        return self.dfs(sources, targets)

    def same_color(self, cell1:hexpex.Axial, cell2:hexpex.Axial) -> bool:
        return self.cell_color(cell1) == self.cell_color(cell2)

    def dfs(self, sources:list[hexpex.Axial], targets:set[hexpex.Axial]) -> bool:
        """ True if any of targets is reachable from any of sources """
        if len(targets) == 0 or len(sources) == 0:
            return False
        visited = set()
        while len(sources) != 0:
            curr = sources.pop()
            if curr in targets:
                return True
            visited.add(curr)

            for neib in curr.ring(1):
                if neib not in visited and\
                        self.is_in_board(neib) and\
                        self.same_color(curr, neib):
                    sources.append(neib)
        return False

    def __str__(self) -> str:
        lines = []
        size = self.position.shape[0]
        for row in range(size):
            lines.append(' ' * row + ' '.join(list(map(lambda i: self.smap[i], self.position[row]))))
        return '\n'.join(lines)

    @staticmethod
    def position_as_str(pos:np.ndarray) -> str:
        """ Console image of a position """
        smap = { key: val for key, val in zip([1, -1, 0], 'xo.') }
        lines = []
        size = pos.shape[0]
        for row in range(size):
            lines.append(' ' * row + ' '.join(list(map(lambda i: smap[i], pos[row]))))
        return '\n'.join(lines)

    def large_print(self) -> str:
        size = self.position.shape[0]
        total_lines = 1 + size * 4 + 2 * (size - 1)
        lines = ['       ' for _ in range(total_lines)]

        lines[0] = '  _____'
        for row in range(size):
            color = self.smap[self.position[row, 0]]
            irow = 1 + 4 * row
            lines[irow + 0] = ' /     \ '
            lines[irow + 1] = f'/   {color}   \\'
            lines[irow + 2] = '\       /'
            lines[irow + 3] = ' \_____/ '

        for col in range(1, size):
            lines[2 * col] += '_____'
            lines[2 * col + 4 * size - 1] += '\ '
            lines[2 * col + 4 * size] += ' \\'
            for idx in range(2 * col + 4 * size + 1, total_lines):
                lines[idx] += '       '
            for row in range(size):
                color = self.smap[self.position[row, col]]
                irow = 1 + 2 * col + 4 * row
                lines[irow + 0] += '     \ '
                lines[irow + 1] += f'  {color}   \\'
                lines[irow + 2] += '      /'
                lines[irow + 3] += '_____/ '

        return '\n'.join(lines)

#    _____
#   /     \
#  /   x   \_____
#  \       /     \
#   \_____/   x   \_____
#   /     \       /     \
#  /   x   \_____/   x   \
#  \       /     \       /
#   \_____/   x   \_____/ 
#   /     \       /     \ 
#  /   x   \_____/   x   \
#  \       /     \       /
#   \_____/   x   \_____/ 
#         \       /     \ 
#          \_____/   x   \
#                \       /
#                 \_____/

def random_game(size:int):
    """ Move by move random game """
    board = HexDriver(size=size)
    for _ in range(size**2):
        board.make_random_move()

def benchmark(niter:int=100):
    """ Random game generation performance test """
    for size in [7, 9, 11, 13]:
        time = timeit.timeit(lambda: random_game(size), number=niter)
        print(f'{time / niter * 1000:.2f} ms per random {size}x{size} game')

def main():
    """ Test program """
    size = 13
    xwin, owin = 0, 0
    for iter in range(1000):
        if iter % 100 == 0:
            print(f'x: {xwin:3d}, o: {owin:3d}')
        board = HexDriver(size=size)
        for _ in range(size**2):
            board.make_random_move()
        assert board.game_over
        if board.red_moves():
            xwin += 1
        else:
            owin += 1
    print(f'x: {xwin}, o: {owin}')

if __name__ == '__main__':
    main()
