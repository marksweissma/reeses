from collections import defaultdict
import variants

from sklearn.base import clone
from sklearn.ensemble import BaseEnsemble
import numpy as np

from .utils import _build_sample_weight


def _parallel_fit_prediction_models(blank, Xs, ys, sample_weights, models, **kwargs):
    assert sorted(Xs) == sorted(ys)

    for leaf in Xs:
        _X = np.vstack(Xs[leaf])
        _y = np.vstack(ys[leaf])
        _sample_weight = np.vstack(sample_weights[leaf]) if sample_weights[leaf] else None

        model.fit(_X, _y, sample_weight=_sample_weight)
        models[leaf] = model
    return models


def _parallel_fit_assigner_leaves(assigner, piece_wise, X, y, sample_weight=None, n_samples_bootstrap=None, **kwargs):
    class_weight = getattr(self.assignment_estimator, 'class_weight', None)

    _sample_weight = _build_sample_weight(assigner, piece_wise,
                                          X, y,
                                          sample_weight,
                                          class_weight=class_weight,
                                          n_samples_bootstrap=n_samples_bootstrap
                                          )

    groups = assigner.apply(X)

    Xs = defaultdict(list)
    ys = defaultdict(list)
    sample_weights = defaultdict(list)
    for group, row, target in zip(groups, X, y):
        Xs[group].append(row)
        ys[group].append(target)

    models = _parallel_fit_prediction_models(piece_wise.prediction_estimator, Xs, ys, sample_weights, **kwargs)

    return {assigner: models}


def fit_controller(piece_wise, X, y=None, n_jobs=1, sample_weight=None, n_samples_bootstrap=None, **kwargs):
    # iterate assignment estimators
    return piece_wise


