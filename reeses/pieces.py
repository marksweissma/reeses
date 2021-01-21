from joblib import Parallel
import threading

from typing import Any

import attr
import numpy as np

from sklearn.base import BaseEstimator, clone, RegressorMixin, ClassifierMixin

from sklearn.utils.fixes import delayed, _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted, _check_sample_weight

from sklearn.ensemble import _forest as sk_forest

from .utils import _build_sample_weight, GroupAssignment
from .defaults import _fallback_dispatch, _get_assignment


def _parallel_fit_prediction_model(prediction_blank, fallback_blank, X, y, prediction_kwargs, sample_weight=None, **kwargs):
    kwargs.update(prediction_kwargs)
    try:
        blank = prediction_blank.fit(X, y, sample_weight=sample_weight, **kwargs)
    except Exception as e:
        if fallback_blank:
            blank = fallback_blank.fit(X, y, sample_weight=sample_weight, **kwargs)
        else:
            raise(e)
    return blank


def _enforce_shape(arr, class_map, classes):
    output = np.zeros((len(arr), len(class_map)))
    for column, _class in enumerate(classes):
        output[:, class_map[_class]] += arr[:, column]
    return output

@attr.s(auto_attribs=True)
class PiecewiseBase(BaseEstimator):
    """
    Backbone for linking assignment estimator to prediction estimators

    Assignment estimator specifies which group an observation belongs.
    Each group has an independent :py:func:`sklearn.base.clone` of the prediction_estimator fit to it

    Bootstrapping supported through the random_state of each estimator ensemble via sample reweighting
    """
    assignment_estimator: BaseEstimator
    prediction_estimator: BaseEstimator

    fallback_estimator: BaseEstimator = attr.ib()
    @fallback_estimator.default
    def _fallback_dispatch(self):
        return _fallback_dispatch(self)

    assignment_method: str = attr.ib(validator=attr.validators.instance_of(str))
    @assignment_method.default
    def _get_assignment(self):
        return _get_assignment(self)

    n_jobs: int = 1
    random_state: Any = None
    verbose = False

    def _make_prediction_estimator(self):
        estimator = clone(self.prediction_estimator)
        return estimator

    def _make_fallback_estimator(self):
        estimator = clone(self.fallback_estimator)
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

            assignments = GroupAssignment.from_model(
                assigner, method=self.assignment_method, X=X, y=y, sample_weight=_sample_weight)

            prediction_template = {assignment: self._make_prediction_estimator() for assignment in assignments.group_ids}
            fallback_template = {assignment: self._make_fallback_estimator() for assignment in assignments.group_ids}

            fitted = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                              **_joblib_parallel_args(prefer='threads'))(
                delayed(_parallel_fit_prediction_model)(
                    prediction_blank=prediction_template[assignment],
                    fallback_blank=fallback_template[assignment],
                    # verbose=self.verbose,
                    prediction_kwargs=prediction_kwargs,
                    **assignments.package_group(assignment)
                )
                for assignment in assignments.group_ids)

            models[assigner] = {group_id: _fitted for group_id,
                                _fitted in zip(assignments.group_ids, fitted)}

        self.shapes = assignments.shapes
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
        self._fit_prediction(X, y=y, n_samples_bootstrap=n_samples_bootstrap,
                             sample_weight=sample_weight, **prediction_kwargs)

        return self

    @property
    def assigners(self):
        # need to check is fitted
        if hasattr(self.assignment_estimator, 'estimators_'):
            assigners = list(self.assignment_estimator.estimators_)
        else:
            assigners = [self.assignment_estimator]

        return assigners

    @property
    def n_assigners(self):
        return len(self.assigners)

    @property
    def n_estimators(self):
        n_estimators = sum([len(assigner_models) for assigner_models in self.estimators_.values()])
        return n_estimators

    @property
    def classes_(self):
        return getattr(self.assignment_estimator, 'classes_', None)

    @property
    def n_classes_(self):
        return getattr(self.assignment_estimator, 'n_classes_', None)


class PiecewiseRegressor(RegressorMixin, PiecewiseBase):
    def predict(self, X):
        predictions = [self._predict(X, assigner) for assigner in self.assigners]
        return np.c_[tuple(predictions)].mean(axis=1)

    def _predict(self, X, assigner):
        assignments = GroupAssignment.from_model(assigner, method=self.assignment_method, X=X)

        group_predictions = {}
        for group in assignments.group_ids:
            model = self.estimators_[assigner][group]
            group_predictions[group] = model.predict(**assignments.package_group(group))

        predictions = assignments.reconstruct_from_groups(group_predictions, shape=self.shapes['y'])
        return predictions


def _predict_proba(piecewise, assigner, X, all_probabilities, lock, models, class_map):
    assignments = GroupAssignment.from_model(assigner, method=piecewise.assignment_method, X=X)

    group_predictions = {}
    for group in assignments.group_ids:
        model = models[group]
        _predictions = model.predict_proba(**assignments.package_group(group))
        group_predictions[group] = _enforce_shape(_predictions, class_map, model.classes_)

    shape = list(piecewise.shapes['y'])
    if len(shape) < 2:
        shape += [piecewise.n_classes_]
    else:
        shape[1] = piecewise.n_classes_

    predictions = assignments.reconstruct_from_groups(group_predictions, shape=shape)
    with lock:
        if len(all_probabilities) == 1:
            all_probabilities[0] += predictions
        else:
            for i in range(len(all_probabilities)):
                all_probabilities[i] += prediction[i]

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
        n_jobs, _, _ = sk_forest._partition_estimators(len(self.assigners), self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_probabilities = [np.zeros((X.shape[0], j), dtype=np.float64)
                     for j in np.atleast_1d(self.n_classes_)]
        class_map = {_class: idx for idx, _class in enumerate(self.classes_)}

        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_predict_proba)(self, assigner, X, all_probabilities,
                                    lock, self.estimators_[assigner], class_map)
            for assigner in self.assigners)

        for probability in all_probabilities:
            probability /= self.n_assigners

        if len(all_probabilities) == 1:
            all_probabilities = all_probabilities[0]

        return all_probabilities

    def predict_log_proba(self, X):
        probabilities = self.predict_proba(X)
        if isinstance(probabilities, list):
            probabilities = [np.log(_probabilities) for _probabilities in probabilities]
        else:
            probabilities = np.log(probabilities)
        return probabilities
