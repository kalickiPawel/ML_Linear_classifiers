from training_models import LinBase
import numpy as np
import time


class Perceptron(LinBase):
    def __init__(self, break_time=10, **kwargs):
        super(Perceptron, self).__init__(**kwargs)
        self.break_time_ = break_time

    def fit(self, X, y):
        m, n = X.shape
        self.class_labels_ = np.unique(y)
        w = self.coef_ if self.coef_ is not None else np.zeros(n)
        b = self.intercept_ if self.intercept_ is not None else 0

        i, k = 0, 0
        t1 = time.time()
        t2 = time.time()
        while k < m and t2 - t1 < self.break_time_:
            if (y[i] * (w.dot(X[i, :]) + b)) <= 0:
                w = w + X[i, :] * y[i]
                b = b + y[i]
                k = 0
            else:
                k += 1
            i = (i + 1) % m
            t2 = time.time()
        self.coef_ = w
        self.intercept_ = b