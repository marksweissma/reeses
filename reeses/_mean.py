from sklearn.base import BaseEstimator
import numpy as np


class MeanEstimator(BaseEstimator):
    def fit(self, X, y=None, **fit_params):
        self.mu_ = y.mean()
        return self

    def predict(self, X):
        output = np.ones((len(X),)) * self.mu_
        return output
