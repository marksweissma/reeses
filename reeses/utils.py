import attr
import inspect
from functools import lru_cache

from collections import defaultdict
from typing import Any, List, Dict
from warnings import catch_warnings, simplefilter, warn

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import check_random_state, check_array, compute_sample_weight

from sklearn.ensemble import _forest as sk_forest


###############################################################
### Several of the helpers are lightly edited sklearn utils ###
###############################################################


def _collect_leaf_indices(tree):
    children_left = tree.children_left
    children_right = tree.children_right
    node_depth = [0] * tree.node_count

    leaves = []

    stack = [(0, 0)]
    while len(stack):
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        is_leaf = children_left[node_id] == children_right[node_id]
        if is_leaf:
            leaves.append(node_id)
        else:
            _depth = depth + 1
            stack.append((children_left[node_id], _depth))
            stack.append((children_right[node_id], _depth))

    return leaves


def _build_sample_weight(tree, reese, X, y, sample_weight, class_weight=None, n_samples_bootstrap=None):
    if getattr(reese.assignment_estimator, 'bootstrap', None):
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = sk_forest._generate_sample_indices(
            tree.random_state, n_samples, n_samples_bootstrap)

        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == 'subsample':
            with catch_warnings():
                simplefilter('ignore', DeprecationWarning)
                curr_sample_weight *= compute_sample_weight('auto', y, indices=indices)

        elif class_weight == 'balanced_subsample':
            curr_sample_weight *= compute_sample_weight('balanced', y, indices=indices)

        _sample_weight = curr_sample_weight
    else:
        _sample_weight = sample_weight
    return _sample_weight


def get_shapes(dict_arr):
    """
    get `shape` attribute of values of dict if they are None and have length

    Args:
        dict_arr (dict): dict of arrays

    Returns:
        dict: keys of `dict_array` with lengths

    """
    return {key: value.shape for key, value in dict_arr.items() if (value is not None and len(value))}


def package_arrays(arrs, shape):
    """
    get `shape` attribute of values of dict if they are None and have length

    Args:
        shape (dict): shape of

    Returns:
        dict: keys of `dict_array` with lengths

    """
    arr = np.vstack(arrs).reshape((len(arrs), *shape[1:]))
    return arr


@attr.s(auto_attribs=True, frozen=True)
class GroupAssignment:
    """
    Serves and reconstructs group data
    """
    data: Dict
    groups: List
    group_ids: List
    shapes: Dict

    @classmethod
    def from_groups(cls, groups, **kwargs):
        """
        Collects **kwargs** data into groups by key specified from  :py:class:`typing.Iterable` of groups
        """
        shapes = get_shapes(kwargs)
        data = {key: defaultdict(list) for key in kwargs if kwargs[key] is not None}
        groups = list(groups)
        group_ids = sorted(set(groups))
        for key, array in filter(lambda x: x[1] is not None, kwargs.items()):
            _data = data[key]
            [_data[group].append(row) for group, row in zip(groups, array)]

        return cls(data, groups, group_ids, shapes)

    @classmethod
    def from_model(cls, model, method, X, **kwargs):
        """
        Builds groups from model by executing method on X, groups X and all **kwargs**
        """
        groups = getattr(model, method)(X)
        return cls.from_groups(groups, X=X, **kwargs)

    def reconstruct_from_groups(self, dict_arr, shape):
        """
        Reconstruct and coerce to shape collection key by group
        """
        output = []
        indices = {leaf: 0 for leaf in self.group_ids}

        for group in self.groups:
            record = dict_arr[group][indices[group]]
            output.append(record)
            indices[group] += 1

        return package_arrays(output, shape)

    def __getitem__(self, key):
        return self.data[key]

    def package_group(self, group, keys=None):
        """
        Serve the stored for the group given keys (None / Falsey yields all)
        """
        keys = key if keys else self.data.keys()
        package = {key: package_arrays(_data[group], self.shapes[key])
                   for key, _data in self.data.items() if key in keys}
        return package


class Pipeline(Pipeline):
    """
    Pipeline wrapper for assignment estimators with apply method
    """
    @if_delegate_has_method(delegate='_final_estimator')
    def apply(self, X):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].apply(Xt)


class MeanEstimator(BaseEstimator):
    """
    Fits on average of y given sample weight (if applicable)
    This is the default in :py:class:`reeses.pieces.PiecewiseBase`

    """
    def fit(self, X, y=None, sample_weight=None, strategy='ovr', **fit_params):
        sample_weight = np.ones_like(y) if sample_weight is None else sample_weight
        count = sample_weight.sum()

        self.mu_ = (sample_weight * y).sum() / count
        return self

    def predict(self, X):
        output = np.ones((len(X), *self.mu_.shape[1:])) * self.mu_
        return output


class MeanRegressor(RegressorMixin, MeanEstimator):
    pass


class MeanClassifier(ClassifierMixin, MeanEstimator):
    """
    Fits on average of y given sample weight (if applicable)
    This is the default in :py:class:`reeses.pieces.PiecewiseBase`

    """
    def fit(self, X, y=None, sample_weight=None, strategy='ovr', **fit_params):

        self.classes_ = np.sort(np.unique(y))
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = y.shape[1] if y.ndim > 1 else 1

        sample_weight = np.ones_like(y) if sample_weight is None else sample_weight
        count = sample_weight.sum()
        groups = GroupAssignment.from_groups(y, weight=sample_weight)
        weights = {group: sum(weight) / count for group, weight in groups['weight'].items()}

        self.mu_ = np.array([weights[group] for group in self.classes_])
        return self

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
        output = np.ones((len(X), *self.mu_.shape)) * self.mu_
        return output

    def predict_log_proba(self, X):
        output = np.log(self.predict_proba(X))
        return output
