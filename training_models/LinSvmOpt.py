import cvxopt
import numpy as np
from training_models import LinBase


class LinSvmOpt(LinBase):
    ksi = []

    def __init__(self, c=1, **kwargs):
        super(LinSvmOpt, self).__init__(**kwargs)
        self.svm_inds_ = None
        self.c_ = c

    def fit(self, X, y):
        m, n = X.shape
        p = np.diag(np.concatenate((np.ones(n), np.zeros(m+1))))
        q = np.zeros(n+1+m)
        q[-m:] = self.c_

        g1 = (-1)*np.hstack((X, np.ones((m, 1))))*np.outer(y, np.ones(n+1))
        g = np.vstack((np.hstack((g1, -np.diag(np.ones(m)))),
                      np.hstack((np.zeros((m, n+1)), -np.diag(np.ones(m))))))
        h = np.vstack(((-1) * np.ones((m, 1)), 0 * np.ones((m, 1))))

        pc = cvxopt.matrix(p)
        qc = cvxopt.matrix(q)
        gc = cvxopt.matrix(g)
        hc = cvxopt.matrix(h)

        solve = cvxopt.solvers.qp(pc, qc, gc, hc)

        self.coef_ = np.array(solve['x'])[0:n, 0]
        self.intercept_ = np.array(solve['x'])[n, 0]
        self.ksi = np.array(solve['x'])[n+1:, 0]
        self.svm_inds_ = np.nonzero(self.ksi >= 1e-4)[0]
        # limit the outline proportionally to the ksi value
        # do not fulfill -> greather
        # self.svm_inds_ = np.nonzero(np.array(solve['z']))[0]

        # compare with sqlearnem
