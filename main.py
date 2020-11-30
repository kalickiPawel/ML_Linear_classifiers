import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import time


class LinBase(BaseEstimator, ClassifierMixin):
    def __init__(self, coef=None, intercept=None, class_labels=None):
        self.coef_ = coef
        self.intercept_ = intercept
        self.class_labels_ = class_labels

    def margin(self, X, y):
        m = y*(X.dot(self.coef_)+self.intercept_)
        return np.min(m), m

    def decision_function(self, X):
        a, b = self.margin(X, np.ones(X.shape[0]))
        return b

    def predict_proba(self, X):
        a, b = self.margin(X, np.ones((X.shape[0],)))
        return np.array([1 - 1 / (1 + np.exp(-b)), 1 / (1 + np.exp(-b))]).T

    def predict(self, X):
        results = np.sign(X.dot(self.coef_) + self.intercept_)
        results_mapped = self.class_labels_[1 * (results > 0)]
        return results_mapped

    def get_params(self, deep=True):
        pass

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __str__(self):
        return "{} [ w={}, b={} ] ".format(self.__class__.__name__, self.coef_, self.intercept_)


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


def plotClass(X, y, clf=None, marker=["ro", "k+"], n=50):
    uy = np.unique(y)
    for i, u in enumerate(uy):
        plt.plot(X[y == u, 0], X[y == u, 1], marker[i], label=str(u))

    if clf is not None:
        x1, x2 = np.meshgrid(np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), n),
                             np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), n))
        xx = np.array([x1.ravel(), x2.ravel()]).T
        # yy = clf.predict_proba(xx)[:, 1]
        yy = clf.predict(xx)
        yy = np.reshape(yy, x1.shape)
        plt.contour(x1, x2, yy, [0.5])
    else:
        pass
    plt.show()


if __name__ == "__main__":
    m, n = 512, 2
    X = np.random.rand(m, n)
    y = (X.dot([1, 1]) > 1) * 2 - 1

    for i, clf in enumerate([
        Perceptron(),
        AveragedPerceptron(),
        VotedPerceptron()
    ]):
        clf.fit(X, y)
        plotClass(X, y, clf)
        print(clf.coef_, clf.intercept_)
