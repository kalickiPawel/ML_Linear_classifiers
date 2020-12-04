import cvxopt
import numpy as np
from training_models import LinBase


class SVM(LinBase):
    def __init__(self, **kwargs):
        super(SVM, self).__init__(**kwargs)
        self.svm_inds_ = None

    def fit(self, X, y):
        m, n = X.shape
        p = np.diag(np.ones(n+1))
        p[-1, -1] = 0
        q = np.zeros(n+1)
        g = (-1)*np.hstack((X, np.ones((m, 1))))*np.outer(y, np.ones(n+1))
        h = (-1)*np.ones((m, 1))

        pc = cvxopt.matrix(p)
        qc = cvxopt.matrix(q)
        gc = cvxopt.matrix(g)
        hc = cvxopt.matrix(h)

        solve = cvxopt.solvers.qp(pc, qc, gc, hc)
        print(solve)

        self.coef_ = np.array(solve['x'])[0:-1, 0]
        self.intercept_ = np.array(solve['x'])[-1, 0]
        self.svm_inds_ = np.nonzero(np.array(solve['z']))[0]
