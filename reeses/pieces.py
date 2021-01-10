from joblib import Parallel, delayed
import threading

from collections import defaultdict
from typing import Any, Callable, Dict
from queue import Queue
from warnings import catch_warnings, simplefilter, warn

import attr
import numpy as np
import threading
import variants

from sklearn.base import BaseEstimator, clone, RegressorMixin, ClassifierMixin

from sklearn.utils import check_random_state, check_array, compute_sample_weight
from sklearn.utils.fixes import delayed, _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted, _check_sample_weight

from sklearn.ensemble import _forest as sk_forest

from .utils import _build_sample_weight, GroupAssignment, get_shape


def _accumulate_prediction(predict, X, out, lock, **kwargs):
    prediction = predict(X, check_input=False, **kwargs)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i, _ in enumerate(out):
                out[i] += prediction[i]


def _parallel_fit_prediction_model(blank, X, y, prediction_kwargs, sample_weight=None, **kwargs):
    kwargs.update(prediction_kwargs)
    blank.fit(X, y, sample_weight=sample_weight, **kwargs)
    return blank



@attr.s(auto_attribs=True)
class PiecewiseBase(BaseEstimator):
    assignment_estimator: BaseEstimator
    prediction_estimator: BaseEstimator
    n_jobs: int = 1
    random_state: Any = None
    verbose=False

    def _make_estimator(self):
        estimator = clone(self.prediction_estimator)
        return estimator

    def _fit_assignment(self, X, y=None, **assignment_kwargs):
        self.assignment_estimator.fit(X, y=y, **assignment_kwargs)

    def _fit_prediction(self, X, y=None, sample_weight=None, n_samples_bootstrap=None, **prediction_kwargs):

        class_weight = getattr(self.assignment_estimator, 'class_weight', None)

        models = {}
        for assigner in self.assigners:
            _sample_weight = _build_sample_weight(assigner, self,
                                                  X, y,
                                                  sample_weight=sample_weight,
                                                  class_weight=class_weight,
                                                  n_samples_bootstrap=n_samples_bootstrap
                                                  )

            assignments = GroupAssignment.from_model(assigner, X=X, y=y, sample_weight=_sample_weight)
            self.shapes = assignments.shapes

            template = {assignment: self._make_estimator() for assignment in assignments.group_ids}

            fitted = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             **_joblib_parallel_args(prefer='threads'))(
                delayed(_parallel_fit_prediction_model)(
                    blank=template[assignment],
                    # verbose=self.verbose,
                    prediction_kwargs=prediction_kwargs,
                    **assignments.package_group(assignment)
                )
                for assignment in assignments.group_ids)

            models[assigner] = {group_id: _fitted for group_id, _fitted in zip(assignments.group_ids, fitted)}


        self.estimators_ = models

    def fit(self, X, y=None, assignment_kwargs=None, prediction_kwargs=None, sample_weight=None, **kwargs):

        X, y = self._validate_data(X, y, multi_output=True,
                                   accept_sparse="csc", dtype=sk_forest.DTYPE)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        self.n_features_ = X.shape[1]

        if y.ndim > 1:
            self.n_outputs_ = y.shape[1]
        elif y.shape[0]:
            self.n_outputs_ = 1
        else:
            raise ValueError("y doesn't have shape")

        max_samples = getattr(self.assignment_estimator, 'max_samples', None)

        n_samples_bootstrap = sk_forest._get_n_samples_bootstrap(
            n_samples=X.shape[0],
            max_samples=max_samples
        )

        assignment_kwargs = assignment_kwargs if assignment_kwargs else {}
        self._fit_assignment(X, y=y, **assignment_kwargs)

        prediction_kwargs = prediction_kwargs if prediction_kwargs else {}
        self._fit_prediction(X, y=y, n_samples_bootstrap=n_samples_bootstrap, sample_weight=sample_weight, **prediction_kwargs)

        return self


    @property
    def n_estimators(self):
        count = len(getattr(self, 'estimators_', []))
        return count

    @property
    def assigners(self):
        # need to check is fitted
        if hasattr(self.assignment_estimator, 'estimators_'):
            assigners = list(self.assignment_estimator.estimators_)
        else:
            assigners = [self.assignment_estimator]

        return assigners

    @property
    def n_estimators(self):
        n_estimators = sum([len(_models) for _models  in self.estimators_])
        return n_estimators



class PiecewiseRegressor(RegressorMixin, PiecewiseBase):
    def predict(self, X):
        predictions = [self._predict(X, assigner) for assigner in self.assigners]
        return np.c_[tuple(predictions)].mean(axis=1)


    def _predict(self, X, assignment_estimator):
        assignments = GroupAssignment.from_model(assignment_estimator, X=X)

        group_predictions = {}
        for group in assignments.group_ids:
            model = self.estimators_[assignment_estimator][group]
            group_predictions[group] = model.predict(**assignments.package_group(group))

        predictions = assignments.reconstruct_from_groups(group_predictions, shape=self.shapes['y'])
        return predictions


class PiecewiseClassifier(ClassifierMixin, PiecewiseBase):
    def predict(self, X):
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            predictions = self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            # all dtypes should be the same, so just take the first
            class_type = self.classes_[0].dtype
            predictions = np.empty((n_samples, self.n_outputs_),
                                   dtype=class_type)

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(np.argmax(proba[k],
                                                                    axis=1),
                                                          axis=0)

        return predictions

    def predict_proba(self, X):
        sk_forest.check_is_fitted(self)
        # Check data
        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [np.zeros((X.shape[0], j), dtype=np.float64)
                     for j in np.atleast_1d(self.n_classes_)]
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(accumulate_prediction)(e.predict_proba, X, all_proba,
                                            lock)
            for e in self.estimators_)

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            all_proba = all_proba[0]

        return all_proba

    def predict_log_proba(self, X):
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba
