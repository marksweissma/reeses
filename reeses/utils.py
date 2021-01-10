import attr
import inspect
from functools import lru_cache

from collections import defaultdict
from typing import Any, List, Dict
from warnings import catch_warnings, simplefilter, warn

import numpy as np

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
    return {key: value.shape for key, value in dict_arr.items() if (value is not None and len(value))}


def package_arrays(arrs, shape):
    arr = np.vstack(arrs).reshape((len(arrs), *shape[1:]))
    return arr


@attr.s(auto_attribs=True, frozen=True)
class GroupAssignment:
    data: Dict
    groups: List
    group_ids: List
    shapes: Dict

    @classmethod
    def from_groups(cls, groups, **kwargs):
        shapes = get_shapes(kwargs)
        data = {key: defaultdict(list) for key in kwargs if kwargs[key] is not None}
        groups = list(groups)
        group_ids = sorted(set(groups))
        for key, array in filter(lambda x: x[1] is not None, kwargs.items()):
            _data = data[key]
            [_data[group].append(row) for group, row in zip(groups, array)]

        return cls(data, groups, group_ids, shapes)

    @classmethod
    def from_model(cls, model, X, **kwargs):
        groups = model.apply(X)
        return cls.from_groups(groups, X=X, **kwargs)

    def reconstruct_from_groups(self, dict_arr, shape):
        output = []
        indices = {leaf: 0 for leaf in self.group_ids}

        for group in self.groups:
            record = dict_arr[group][indices[group]]
            output.append(record)
            indices[group] += 1

        return package_arrays(output, shape)

    def __getitem__(self, key):
        return self.data[key]

    def package_group(self, group):
        package = {key: package_arrays(_data[group], self.shapes[key])
                   for key, _data in self.data.items()}
        return package


class Pipeline(Pipeline):
    @if_delegate_has_method(delegate='_final_estimator')
    def apply(self, X):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].apply(Xt)
