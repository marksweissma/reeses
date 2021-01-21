from functools import singledispatch
from sklearn.base import RegressorMixin, ClassifierMixin
from .utils import MeanEstimator, MeanClassifier, MeanRegressor

@singledispatch
def _fallback_dispatch(piecewise):
    model = MeanEstimator()
    return model

@_fallback_dispatch.register(RegressorMixin)
def _fallback_dispatch_regressor(piecewise):
    model = MeanRegressor()
    return model

@_fallback_dispatch.register(ClassifierMixin)
def _fallback_dispatch_classifier(piecewise):
    model = MeanClassifier()
    return model

def _get_assignment(piecewise):
    name = None
    if hasattr(piecewise.assignment_estimator, 'apply'):
        name = 'apply'
    elif hasattr(piecewise.assignment_estimator, 'predict'):
        name = 'predict'
    return name

