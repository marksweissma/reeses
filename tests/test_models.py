import pytest

import numpy.testing as npt

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston, load_iris
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score

from singledispatch import singledispatch
from reeses import PiecewiseRegressor, PiecewiseClassifier, MeanRegressor, MeanClassifier

RANDOM_STATE = 42

def tree_classifier():
    return DecisionTreeClassifier(min_samples_leaf=50, random_state=RANDOM_STATE)

def tree_regressor():
    return DecisionTreeRegressor(min_samples_leaf=50, random_state=RANDOM_STATE)


def forest_classifier():
    return RandomForestClassifier(n_estimators=20, min_samples_leaf=50, random_state=RANDOM_STATE, n_jobs=-1)

def forest_regressor():
    return RandomForestRegressor(n_estimators=20, min_samples_leaf=50, random_state=RANDOM_STATE, n_jobs=-1)


def logit():
    return LogisticRegression()

def linear():
    return LinearRegression()


def boston():
    return load_boston()

def iris():
    return load_iris()


@singledispatch
def build_piecewise(assigner, prediction):
    raise TypeError('assigner not classifier or regressor')

@build_piecewise.register(RegressorMixin)
def build_piecewise_regressor(assigner, prediction):
    return PiecewiseRegressor(assigner, prediction)

@build_piecewise.register(ClassifierMixin)
def build_piecewise_classifier(assigner, prediction):
    return PiecewiseClassifier(assigner, prediction)

@singledispatch
def score(piecewise, X_test, y_test, equivalence):
    raise TypeError('piecewise not classifier or regressor')

@score.register(RegressorMixin)
def score_regressor(piecewise, X_test, y_test, equivalence):
    predictions = piecewise.predict(X_test)
    assignment_predictions = piecewise.assignment_estimator.predict(X_test)

    if equivalence:
        npt.assert_almost_equal(predictions, assignment_predictions)
    else:
        score = r2_score(y_test, predictions)
        assignment_score = r2_score(y_test, assignment_predictions)
        assert score > assignment_score


@score.register(ClassifierMixin)
def score_classifier(piecewise, X_test, y_test, equivalence):
    probabilities = piecewise.predict_proba(X_test)
    assignment_probabilities = piecewise.assignment_estimator.predict_proba(X_test)

    values = piecewise.predict(X_test)
    assignment_values = piecewise.assignment_estimator.predict(X_test)

    if equivalence:
        npt.assert_almost_equal(probabilities, assignment_probabilities)
        npt.assert_almost_equal(values, assignment_values)
    else:
        score = roc_auc_score(y_test, probabilities, multi_class='ovr')
        assignment_score = roc_auc_score(y_test, assignment_probabilities, multi_class='ovr')
        assert score > assignment_score

        score = accuracy_score(y_test, values)
        assignment_score = accuracy_score(y_test, assignment_values)
        assert score > assignment_score


@pytest.mark.parametrize("assigner,prediction,dataset",
                         [(tree_classifier(), logit(), load_iris()),
                          (tree_regressor(), linear(), load_boston()),
                          (forest_classifier(), logit(), load_iris()),
                          (forest_regressor(), linear(), load_boston()),
                          ]
                         )
def test_scoring(assigner, prediction, dataset):
    piecewise = build_piecewise(assigner, prediction)

    X = dataset['data']
    y = dataset['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    piecewise.fit(X_train, y_train)
    score(piecewise, X_test, y_test, False)
