from training_models import Perceptron
import numpy as np
import time


class VotedPerceptron(Perceptron):
    def __init__(self, k_max=1000, **kwargs):
        super(VotedPerceptron, self).__init__(**kwargs)
        self.k_max = k_max
        self.v_ = None
        self.v_bias_ = None
        self.ck_ = np.zeros(self.k_max + 1)

    def fit(self, X, y):
        m, n = X.shape
        self.class_labels_ = np.unique(y)
        self.v_ = np.zeros((self.k_max + 1, n))
        self.v_bias_ = np.zeros(self.k_max + 1)
        w = self.coef_ if self.coef_ is not None else np.zeros(n)
        b = self.intercept_ if self.intercept_ is not None else 0

        i, k = 0, 0
        ck, ci = 0, 0
        t1, t2 = time.time(), time.time()
        while k < self.k_max and t2 - t1 < self.break_time_:
            ci += 1
            if (y[i] * (w.dot(X[i, :]) + b)) <= 0:
                self.v_[k, :] += w
                self.v_bias_[k] += b
                self.ck_[k] = ck
                w = w + X[i, :] * y[i]
                b = b + y[i]
                k += 1
                ck = 1
            else:
                ck += 1
            i = (i + 1) % m
            t2 = time.time()

        self.v_[k, :] += w
        self.v_bias_[k] += b
        self.ck_[k] = ck
        self.coef_ = self.v_
        self.intercept_ = self.v_bias_

    def predict(self, X):
        v = np.hstack((self.v_, np.array([self.v_bias_]).T))
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return np.sign(np.dot(np.sign(np.dot(v, X.T)).T, self.ck_))

    def predict_proba(self, X):
        print(self.v_.shape)
        print(self.v_bias_.shape)
        v = np.hstack((self.v_, np.array([self.v_bias_]).T))
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        a = np.sign(np.dot(np.sign(np.dot(v, X.T)).T, self.ck_))
        b = 1 - a
        return np.vstack(a, b).T