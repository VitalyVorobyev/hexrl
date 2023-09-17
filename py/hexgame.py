""" """

import numpy as np
import hexpex

class HexDriver:
    """ Hex game mechanics """
    def __init__(self, size:int=11):
        """ Ctor """
        self.moves = 0
        self.blue = 1
        self.red = -1
        self.position = np.zeros((size, size), dtype=int)
    
    def __str__(self) -> str:
        smap = {-1: 'x', 0: '.', 1: 'o'}
        size = self.position.shape[0]
        total_lines = size * 5 + 2 * (size - 1)
        lines = ['' for _ in range(total_lines)]

        lines[0] = '  _____'
        for row in range(size):
            irow = 4 * row
            lines[irow + 1] = ' /     \ '
            lines[irow + 2] = '/       \\'
            lines[irow + 3] = '\       /'
            lines[irow + 4] = ' \_____/'

        
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
  
    def make_move(self, q:int, r:int) -> bool:
        """ """
        if not self.is_in_board(q, r) or self.cell_color(q, r) != 0:
            return False
        hsize = self.position.shape[0] // 2
        col, row = q + hsize, r + hsize
        self.position[row, col] = self.red if self.moves % 2 else self.blue
        return True

    def red_moves(self) -> bool:
        """ True if red makes the next move """
        return self.moves % 2

    def is_in_board(self, q:int|hexpex.Cube, r:int|None) -> bool:
        hsize = self.position.shape[0] // 2
        col, row = q.q, q.r if r is None else q, r
        col, row = col + hsize, row + hsize
        return col > 0 and col < size and row > 0 and row < size

    def cell_color(self, q:int|hexpex.Cube, r:int|None) -> int:
        hsize = self.position.shape[0] // 2
        col, row = q.q, q.r if r is None else q, r
        return self.position[row + hsize, col + hsize]
    
    def blue_wins(self) -> bool:
        """ True if a winning position for blue """
        hsize = self.position.shape[0] // 2
        sources = [hexpex.Cube(-hsize, r, hsize - r) for r in range(-hsize, hsize + 2)
                   if self.cell_color(-hsize, r) == self.blue]
        targets = [hexpex.Cube(hsize + 1, r, -hsize - r - 1) for r in range(-hsize, hsize + 2)
                   if self.cell_color(hsize + 1, r) == self.blue]
        return self.dfs(sources, targets)
    
    def red_wins(self) -> bool:
        """ """
        hsize = self.position.shape[0] // 2
        sources = [hexpex.Cube(q, -hsize, hsize - q) for q in range(-hsize, hsize + 2)
                   if self.cell_color(q, -hsize) == self.blue]
        targets = [hexpex.Cube(q, hsize + 1, -hsize - q - 1) for q in range(-hsize, hsize + 2)
                   if self.cell_color(q, hsize + 1) == self.blue]
        return self.dfs(sources, targets)
    
    def same_color(self, cell1:hexpex.Cube, cell2:hexpex.Cube) -> bool:
        return self.cell_color(cell1) == self.cell_color(cell2)

    def dfs(self, sources:list[hexpex.Cube], targets:set[hexpex.Cube]) -> bool:
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


if __name__ == '__main__':
    """ """
    size = 3
    board = HexDriver(size=size)
    print(board)
