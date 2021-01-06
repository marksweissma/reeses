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
from sklearn.ensemble._base import BaseEnsemble

from sklearn.utils import check_random_state, check_array, compute_sample_weight
from sklearn.utils.fixes import delayed, _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted, _check_sample_weight

from sklearn.ensemble import _forest as sk_forest

from .utils import get_variant, _build_sample_weight, GroupAssignment
from .fitting import fit_controller


@attr.s(auto_attribs=True)
class PiecewiseBase(BaseEstimator):
    assignment_estimator: BaseEstimator
    prediction_estimator: BaseEstimator
    n_jobs: int = 1
    random_state: Any = None
    verbose=False

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.prediction_estimator)

        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def _fit_assignment(self, X, y=y, **assignment_kwargs):
        self.assignment_estimator.fit(X, y=y, **assignment_kwargs)

    def _fit_prediction(self, X, y=y, **prediction_kwargs):

        models = {}
        for assigner in self.assigners:
            assignments = GroupAssignment.from_model(assigner, X)

            template = {(assigner, assignment): clone(blank) for assignment in assignments.group_ids}


            fitted = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             **_joblib_parallel_args(prefer='threads'))(
                delayed(_parallel_fit_tree_leaves)(
                    assigner, self, X, y, sample_weight,
                    verbose=self.verbose, class_weight=template.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap)
                for assigner in self.assigners)

            models[assigner] = fitted 

        [_parallel_fit_tree_leaves(assigner, self, X, y, sample_weight, n_samples_bootstrap)
                  for assigner in self.assigners]

        self.estimators_ = models

    def fit(self, X, y=None, assignment_kwargs=None, prediction_kwargs=None, sample_weight=None, **kwargs):

        X, y = self._validate_data(X, y, multi_output=True,
                                   accept_sparse="csc", dtype=sk_forest.DTYPE)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        self.n_features_ = X.shape[1]
        self.n_outputs_ = y.shape[1] if y.ndim > 1 else 1 if y.shape[0] else raise ValueError("y doesn't have shape")

        max_samples = getattr(self.assignment_estimator, 'max_samples', 1.0)

        n_samples_bootstrap = sk_forest._get_n_samples_bootstrap(
            n_samples=X.shape[0],
            max_samples=max_samples
        )

        assignment_kwargs = assignment_kwargs if assignment_kwargs else {}
        self._fit_assignment(X, y=y, **assignment_kwargs)

        prediction_kwargs = prediction_kwargs if prediction_kwargs else {}

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
        n_estimators = sum([len(_models) for _models  in self.prediction_models_])
        return n_estimators


# hack because np.c_[list(elements)] gives the transpose of the desired output
def _array_cat(*arrs):
    return np.c_[arrs]


class PiecewiseRegressor(RegressorMixin, PieceBase):
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


class PiecewiseClassifier(ClassifierMixin, PieceBase):
    def predict(self, X):
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

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
            delayed(_accumulate_prediction)(e.predict_proba, X, all_proba,
                                            lock)
            for e in self.estimators_)

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba

    def predict_log_proba(self, X):
        """
        Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the trees in the
        forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba
