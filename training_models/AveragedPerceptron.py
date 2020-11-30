from training_models import Perceptron
import numpy as np
import time


class AveragedPerceptron(Perceptron):
    def __init__(self, k_max=1000, **kwargs):
        super(AveragedPerceptron, self).__init__(**kwargs)
        self.k_max = k_max
        self.v_ = None
        self.v_bias_ = None

    def fit(self, X, y):
        m, n = X.shape
        self.class_labels_ = np.unique(y)
        w = self.coef_ if self.coef_ is not None else np.zeros(n)
        b = self.intercept_ if self.intercept_ is not None else 0

        self.v_, self.v_bias_ = 0 * w, 0 * b

        i, k = 0, 0
        ck, ci = 0, 0
        t1 = time.time()
        t2 = time.time()
        while k < self.k_max and t2 - t1 < self.break_time_:
            ci += 1
            if (y[i] * (w.dot(X[i, :]) + b)) <= 0:
                self.v_ += ck * w
                self.v_bias_ += ck * b
                w = w + X[i, :] * y[i]
                b = b + y[i]
                k += 1
                ck = 1
            else:
                ck += 1
            i = (i + 1) % m
            t2 = time.time()

        self.v_ += ck * w
        self.v_bias_ += ck * b

        self.coef_ = self.v_
        self.intercept_ = self.v_bias_