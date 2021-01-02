from joblib import Parallel

from collections import defaultdict
from typing import Any, Callable, Dict
from queue import Queue
from warnings import catch_warnings, simplefilter, warn

import attr
import numpy as np
import threading
import variants

from sklearn.base import BaseEstimator, clone, RegressorMixin, ClassifierMixin
from sklearn.ensemble._base import BaseEnsemble

from sklearn.utils import check_random_state, check_array, compute_sample_weight
from sklearn.utils.fixes import delayed, _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted, _check_sample_weight

from sklearn.ensemble import _forest as sk_forest

from .utils import get_variant, _build_sample_weight, GroupAssignment
from .fitting import fit_controller


@variants.primary
def ensemble_reducer(method, predictions):
    return getattr(ensemble_reducer, method)(predictions)

@ensemble_reducer.variant('predict')
def ensemble_reducer(predictions):
    return np.c_[predictions].mean(axis=1)

@ensemble_reducer.variant('predict_proba')
def ensemble_reducer(predictions):
    n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)


@ensemble_reducer.variant('predict_log_proba')
def ensemble_reducer(predictions):
    pass


@attr.s(auto_attribs=True)
class ReeseBase(BaseEstimator):
    assignment_estimator: BaseEstimator
    prediction_estimator: BaseEstimator
    variant: str = None
    n_jobs: int = 1

    def fit(self, X, y=None, assignment_kwargs=None, prediction_kwargs=None, sample_weight=None, **kwargs):

        max_samples = getattr(self.assignment_estimator, 'max_samples', 1.0)

        n_samples_bootstrap = sk_forest._get_n_samples_bootstrap(
            n_samples=X.shape[0],
            max_samples=max_samples
        )

        assignment_kwargs = assignment_kwargs if assignment_kwargs else {}
        prediction_kwargs = prediction_kwargs if prediction_kwargs else {}

        self.assignment_estimator.fit(X, y=y, **assignment_kwargs)

        payload = {'sample_weight': sample_weight,
                   'n_samples_bootstrap': n_samples_bootstrap,
                   'n_jobs': self.n_jobs
                   }
        payload.update(prediction_kwargs)

        fit_controller(self, X, y=y, variant=self.variant, **prediction_kwargs)

        return self

    @property
    def assigners(self):
        if hasattr(self.assignment_estimator, 'estimators_'):
            assigners = [i for i in self.assignment_estimator.estimators_]
        else:
            assigners = [self.assignment_estimator]

        return assigners


# hack because np.c_[list(elements)] gives the transpose of the desired output
def _array_cat(*arrs):
    return np.c_[arrs]


class ReeseRegressor(RegressorMixin, ReeseBase):
    def predict(self, X):
        predictions = [self._predict(X, assigner) for assigner in self.assigners]
        return _array_cat(*predicions).mean(axis=1)


    def _predict(self, X, assignment_estimator):
        leaf_assignments = GroupAssignment.from_model(assignment_estimator, X)

        group_predictions = {}
        for group in leaf_assignments.leaves:
            model = self.prediction_models_[group]
            data = leaf_assignments[group]
            group_predictions[group] = model.predict(data)

        predictions = leaf_assignments.reconstruct_from_groups(group_predictions)
        return predictions


# class ReeseClassifier(ClassifierMixin, ReeseBase):
    # def predict_proba(self, X):
        # return predictions_controller(self, X, method='predict_proba')

    # def predict_log_proba(self, X):
        # return predictions_controller(self, X, method='predict_proba')
