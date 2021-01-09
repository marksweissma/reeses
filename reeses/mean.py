from sklearn.base import BaseEstimator
import numpy as np


class MeanEstimator(BaseEstimator):
    def fit(self, X, y=None, **fit_params):
        self.mu_ = y.mean(axis=1)
        return self

    def predict(self, X):
        if len(shape) > 1:
            _shape = (len(X), *shape[1:])
        return np.ones((len(X), _shape)) * self.mu
