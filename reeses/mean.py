from sklearn.base import BaseEstimator
import numpy as np


class MeanEstimator(BaseEstimator):
    def fit(self, X, y=None, sample_weight=None, **fit_params):
        sample_weight = np.ones_like(y) if sample_weight is None else sample_weight
        count = sample_weight.sum()

        self.mu_ = (sample_weight * y).sum() / count
        return self

    def predict(self, X):
        output = np.ones((len(X),)) * self.mu_
        return output
