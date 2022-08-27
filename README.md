# reeses

reeses is a scikit-learn plugin for piecewise models with learned partitions.

learned partitions or groups can be assigned through tree based or clustering models,
the only requirements is the estimator can be fit on the data and exposes a method
that can assign the appropriate id of the learned partition for new data. prediction
estimators can be optimized within node / leaf through grid search if neccesary 

reeses tries to follow the utility patterns in scikit-learn i.e.

  1. sample weight transformation for bootstrapping
  2. joblib for parallelization


## quick start

``` bash
pip install reeses
```

### regression

```python

from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from reeses import PiecewiseRegressor

data = load_boston()

X = data['data']
y = data['target']

tree = DecisionTreeRegressor(min_samples_leaf=40)
ols = LinearRegression()

model = PiecewiseRegressor(assignment_estimator=tree, prediction_estimator=ols)
model.fit(X, y)
```

### classification

```python

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from reeses import PiecewiseClassifier

data = load_iris()

X = data['data']
y = data['target']

tree = DecisionTreeClassifier(min_samples_leaf=40)
logit = LogisticRegression()

model = PiecewiseClassifier(assignment_estimator=tree, prediction_estimator=logit)
model.fit(X, y)
```

### ensembling & bootstrapping

reeses will introspect ensemble assignment estimators and maintain the bootstrapped sample
fit in each assignment estimator for prediction estimators associated with that assignment estimator. 

```python

from sklearn.datasets import load_iris
from sklearn.tree import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from reeses import PiecewiseClassifier

data = load_iris()

X = data['data']
y = data['target']

forest = RandomForestClassifier(n_estimators=10, min_samples_leaf=40)
logit = LogisticRegression()

model = PiecewiseClassifier(assignment_estimator=forest, prediction_estimator=logit)
model.fit(X, y)
```

### clustering assignment

reeses supports arbitary assignment estimators. scikit-learn clustering estimators
use the `predict` method to make assignments. <`0.0.3` reeses defaults to the `apply` method
but con be configured of any assignment through the `assignment_method` attribute. >=`0.0.3`
defaults to `apply` if assignment estimator has apply else `predict` if has predict else raises.


### Example

Consider `y = abs(x)`. This function is a poor candidate for linear methods (will predict a constant)
and tree based methods (cannot interpolate outside the observed bounds). Combining the two we can
produce an effective estimator.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

from reeses import pieces
```

create data
```python
observations = 10000
x_train = np.hstack([np.random.uniform(0, 1, observations // 2),np.random.uniform(-1, 0, observations // 2)])
x_test = np.random.uniform(-2, 2, observations)
y_train = np.abs(x_train) + norm.rvs(size=observations, loc=0, scale=.05)
y_test = np.abs(x_test) + norm.rvs(size=observations, loc=0, scale=.05)

# reshape arrays to 2D
x_train = x_train[:, None]
x_test = x_test[:, None]
```

![Training Data](/docs/source/images/reeses_train.png)

create models

```python 
models = {
    'ols':
    LinearRegression(),
    'tree':
    DecisionTreeRegressor(max_depth=10),
    'piecewise_tree':
    pieces.PiecewiseRegressor(assignment_estimator=DecisionTreeRegressor(
        max_depth=1, criterion='mse'),
                              prediction_estimator=LinearRegression()),
}

grid_params = {
    'tree': {
        'max_depth': range(1, 20)
    },
    'piecewise_tree': {
        'assignment_estimator__max_depth': range(1, 5)
    }
}
```

fit models and make predictions on test data
```python
predictions = {}
for name, estimator in models.items():
    if name in grid_params:
        predictions[name] = GridSearchCV(
            estimator, grid_params[name],
            n_jobs=-1).fit(x_train, y_train).best_estimator_.predict(x_test)
    else:
        predictions[name] = estimator.fit(x_train, y_train).predict(x_test)
```

![Test Data](/docs/source/images/reeses_test_data.png)

Results! colorbar is magnitude of residuals

![OLS](/docs/source/images/reeses_ols.png)

![Decision Tree](/docs/source/images/reeses_tree.png)

![Piecewise Decision Tree](/docs/source/images/reeses_piecewise_tree.png)


