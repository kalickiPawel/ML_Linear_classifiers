import numpy as np
import matplotlib.pyplot as plt
from training_models import Perceptron, AveragedPerceptron, VotedPerceptron


def plotClass(X, y, clf=None, marker=["ro", "k+"], n=50):
    uy = np.unique(y)
    for i, u in enumerate(uy):
        plt.title(type(clf).__name__)
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

    # X = np.random.rand(m, n)
    # y = (X.dot([1, 1]) > 1) * 2 - 1

    x1 = np.random.rand(m, n)
    x2 = np.random.rand(m, n)
    x2[:, 1] = x2[:, 1] + 0.8
    X = np.vstack((x1, x2))
    y = np.concatenate((-np.ones(m), np.ones(m)))

    for i, clf in enumerate([
        Perceptron(),
        AveragedPerceptron(),
        VotedPerceptron()
    ]):
        clf.fit(X, y)
        plotClass(X, y, clf)
        print(clf.coef_, clf.intercept_)
