""" Union-find data structure implementation """

import numpy as np

class Unionfind:
    """ The union-find data structure """
    def __init__(self, size:int) -> None:
        self.items = np.arange(size)

    def find(self, idx:int) -> int:
        """ Find the root """
        curitem = idx
        curpath = []
        while self.items[curitem] != curitem:
            curpath.append(curitem)
            curitem = self.items[curitem]
        self.items[curpath[:-1]] = curitem
        return curitem

    def union(self, idx1:int, idx2:int) -> None:
        """ Unoin elements """
        self.items[self.find(idx1)] = self.find(idx2)
