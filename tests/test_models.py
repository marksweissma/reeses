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


@pytest.fixture(scope="session")
def datasets():
    output = {'boston': load_boston(), 'iris': load_iris()}
    return output


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

def mean_classifier():
    return MeanClassifier()

def mean_regressor():
    return MeanRegressor()


@singledispatch
def build_piecewise(assigner, prediction):
    raise TypeError('assigner not classifier or regressor')

@build_piecewise.register(RegressorMixin)
def build_piecewise_regressor(assigner, prediction):
    return PiecewiseRegressor(assigner, prediction)

@build_piecewise.register(ClassifierMixin)
def build_piecewise_classifier(assigner, prediction):
    return PiecewiseClassifier(assigner, prediction)

def _get_predictions(model, X, method):
    predictions = getattr(model, method)(X)
    assignment_predictions = getattr(model.assignment_estimator, method)(X)
    return predictions, assignment_predictions


@singledispatch
def get_dataset(piecewise, datasets):
    raise TypeError('piecewise not classifier or regressor')

@get_dataset.register(RegressorMixin)
def get_dataset_regressor(piecewise, datasets):
    return datasets['boston']

@get_dataset.register(ClassifierMixin)
def get_dataset_classifier(piecewise, datasets):
    return datasets['iris']


@singledispatch
def equate(piecewise, X_test, y_test):
    raise TypeError('piecewise not classifier or regressor')

@equate.register(RegressorMixin)
def equate_regressor(piecewise, X_test, y_test):
    response = _get_predictions(piecewise, X_test, 'predict')
    predictions, assignment_predictions = response
    npt.assert_almost_equal(predictions, assignment_predictions)

@equate.register(ClassifierMixin)
def equate_classifier(piecewise, X_test, y_test):
    response = _get_predictions(piecewise, X_test, 'predict_proba')
    probabilities, assignment_probabilities = response

    response = _get_predictions(piecewise, X_test, 'predict')
    values, assignment_values = response

    npt.assert_almost_equal(probabilities, assignment_probabilities)
    npt.assert_almost_equal(values, assignment_values)


@singledispatch
def score(piecewise, X_test, y_test):
    raise TypeError('piecewise not classifier or regressor')

@score.register(RegressorMixin)
def score_regressor(piecewise, X_test, y_test):
    response = _get_predictions(piecewise, X_test, 'predict')
    predictions, assignment_predictions = response
    score = r2_score(y_test, predictions)
    assignment_score = r2_score(y_test, assignment_predictions)
    assert score > assignment_score


@score.register(ClassifierMixin)
def score_classifier(piecewise, X_test, y_test):
    response = _get_predictions(piecewise, X_test, 'predict_proba')
    probabilities, assignment_probabilities = response

    response = _get_predictions(piecewise, X_test, 'predict')
    values, assignment_values = response

    score = roc_auc_score(y_test, probabilities, multi_class='ovr')
    assignment_score = roc_auc_score(y_test, assignment_probabilities, multi_class='ovr')
    assert score > assignment_score

    score = accuracy_score(y_test, values)
    assignment_score = accuracy_score(y_test, assignment_values)
    assert score > assignment_score


@pytest.mark.parametrize("assigner,prediction",
                         [(tree_classifier(), mean_classifier()),
                          (tree_regressor(), mean_regressor()),
                          (forest_classifier(), mean_classifier()),
                          (forest_regressor(), mean_regressor()),
                          ]
                         )
def test_equivalence(assigner, prediction, datasets):
    piecewise = build_piecewise(assigner, prediction)
    dataset = get_dataset(piecewise, datasets)

    X = dataset['data']
    y = dataset['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    piecewise.fit(X_train, y_train)
    equate(piecewise, X_test, y_test)


@pytest.mark.parametrize("assigner,prediction",
                         [(tree_classifier(), logit()),
                          (tree_regressor(), linear()),
                          (forest_classifier(), logit()),
                          (forest_regressor(), linear()),
                          ]
                         )
def test_scoring(assigner, prediction, datasets):
    piecewise = build_piecewise(assigner, prediction)
    dataset = get_dataset(piecewise, datasets)

    X = dataset['data']
    y = dataset['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    piecewise.fit(X_train, y_train)
    score(piecewise, X_test, y_test)
