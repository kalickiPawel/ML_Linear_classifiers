import numpy as np


class LinearData:
    def __init__(self, m, n):
        self.x = np.random.rand(m, n)
        self.y = (self.x.dot([1, 1]) > 1) * 2 - 1


class LinearDataNonSep(LinearData):
    def __init__(self, m, n):
        super().__init__(m, n)
        x1 = np.random.rand(m, n)
        x2 = np.random.rand(m, n)
        x2[:, 1] = x2[:, 1] + 0.8
        self.x = np.vstack((x1, x2))
        self.y = np.concatenate((-np.ones(m), np.ones(m)))
