import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from training_models import Perceptron
from training_models import AveragedPerceptron
from training_models import VotedPerceptron
from training_models import LinSvmOpt
from datasets import *


def plotClass(X, y, data, clf=None, marker=None, n=50):
    if not isinstance(data, DataLoader):
        if marker is None:
            marker = ["ro", "k+"]
        plt.figure()
        uy = np.unique(y)
        for i, u in enumerate(uy):
            plt.title(type(clf).__name__)
            plt.plot(X[y == u, 0], X[y == u, 1], marker[i], label=str(u))

        if clf is not None:
            if isinstance(clf, LinSvmOpt):
                # plt.plot(self.svm_inds, '*')
                pass
                # TODO: plot dla punktów clf.svm_inds
            x1, x2 = np.meshgrid(np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), n),
                                 np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), n))
            xx = np.array([x1.ravel(), x2.ravel()]).T
            yy = clf.predict_proba(xx)[:, 1]
            # yy = clf.predict(xx)
            yy = np.reshape(yy, x1.shape)
            plt.contour(x1, x2, yy, [0.5])
        else:
            pass
        plt.show()


def spiral(p, noise=1):
    t = np.linspace(0, 2*2*np.pi, p)    # 2*np.pi -> kąt pełny
    x1 = np.array([t * np.cos(t), t * np.sin(t)]).T
    x2 = np.array([t * np.cos(t+np.pi), t * np.sin(t+np.pi)]).T
    x = np.vstack((x1, x2))
    x = x + np.random.rand(*x.shape) * noise
    y = np.concatenate((np.ones(p), -np.ones(p)))
    return x, y


if __name__ == "__main__":
    # TODO -> class to create data sets
    #   chess
    #   crescents -> from scikit-learn
    #   nested spirals

    m, n = 512, 2

    # data = LinearData(m, n)
    # data = LinearDataNonSep(m, n)
    data = DataLoader('sonar', 'csv')
    # data = DataLoader('reuters')    # -> not for Perceptrons
    # data = ChessBoard()
    # data = Spiral()
    # data = Circles()

    X = data.x
    y = data.y

    for i, clf in enumerate([
        Perceptron(),
        AveragedPerceptron(),
        VotedPerceptron(),
        # LinSvmOpt()
    ]):
        clf.fit(X, y)
        plotClass(X, y, data, clf)
        print(clf.__str__())
        a, b = clf.margin(X, y)
        print("Wartość marginesu: ", a)

    # x, y = spiral(500)
    # plt.scatter(x[:, 0], x[:, 1], c=y)
    # plt.show()

    # np.rand(()) * 3
    # rand * 3 /
    # where sum is % 2 or not % 2
    # np.sum axis % 2 == 0
    # int or round
