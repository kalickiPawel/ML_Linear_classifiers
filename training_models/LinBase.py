from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


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