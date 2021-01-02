from collections import defaultdict
import variants

from sklearn.base import clone
from sklearn.ensemble import BaseEnsemble
import numpy as np

from .utils import get_variant, _build_sample_weight


def _parallel_fit_prediction_models(reese, Xs, ys, sample_weights, models, **kwargs):
    assert sorted(Xs) == sorted(ys)

    for leaf in Xs:
        _X = np.vstack(Xs[leaf])
        _y = np.vstack(ys[leaf])
        _sample_weight = np.vstack(sample_weights[leaf]) if sample_weights[leaf] else None

        model = clone(reese.prediction_estimator)
        model.fit(_X, _y, sample_weight=_sample_weight)
        models[leaf] = model
    return models

def _parallel_fit_tree_leaves(tree, reese, X, y, sample_weight, n_samples_bootstrap=None, **kwargs):
    class_weight = getattr(self.assignment_estimator, 'class_weight', None)

    _sample_weight = _build_sample_weight(tree, reese,
                                          X, y,
                                          sample_weight,
                                          class_weight=class_weight,
                                          n_samples_bootstrap=n_samples_bootstrap
                                          )

    groups = tree.apply(X)

    Xs = defaultdict(list)
    ys = defaultdict(list)
    sample_weights = defaultdict(list)
    for group, row, target in zip(groups, X, y):
        Xs[group].append(row)
        ys[group].append(target)

    models = _parallel_fit_prediction_models(reese, Xs, ys, sample_weights, **kwargs)

    return models


@variants.primary
def fit_controller(reese, X, y=None, variant=None, ensemble_types=(BaseEnsemble,), n_jobs=1, **kwargs):
    """
    Invoke fitting variant for single vs ensemble assignment estimator estimator
    """
    tree_instance = reese.assignment_estimator
    prediction_instance = reese.prediction_estimator
    variant = get_variant(variant, tree_instance, prediction_instance, ensemble_types)
    reese = getattr(fit_controller, variant)(reese, X, y=y, n_jobs=n_jobs, **kwargs)
    return reese


@fit_controller.variant('single')
def fit_controller(reese, X, y=None, n_jobs=1, **kwargs):
    models = _parallel_fit_tree_leaves(reese.assignment_estimator, reese, X, y, reese.sample_weight, models={})
    reese.prediction_models_ = models
    return reese


@fit_controller.variant('ensemble')
def fit_controller(reese, X, y=None, n_jobs=1, **kwargs):
    models = {idx: {} for idx in reese.assignment_estimator.estimators_}
    [_parallel_fit_tree_leaves(reese.estimators_[idx], reese, X, y, reese.sample_weight, models[idx]) for idx in models]
    reese.prediction_models_ = models
    return reese


