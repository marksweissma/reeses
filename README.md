# reeses

reeses is a scikit-learn plugin for piecewise models with learned partitions.

learned partitions or groups can be assigned through tree based or clustering models,
the only requirements is the estimator can be fit on the data and exposes a method
that can assign the appropriate id of the learned partition for new data.

reeses follows the utility patterns in scikit-learn such as 

  1. sample weight transformation for bootstrapping
  2. joblib for parallelization
  3. threading lock for parallel predictions


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

tree = DecisionTreeRegressor(min_samples_leaf(10))
ols = LinearRegression()

model = PiecewiseRegressor(assignment_estimator=tree, prediction_estimator=ols)
model.fit(X, y)
```

### classification

```python

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from reeses import PiecewiseRegressor

data = load_iris()

X = data['data']
y = data['target']

tree = DecisionTreeClassifier(min_samples_leaf(10))
ols = LogisticRegression()

model = PiecewiseRegressor(assignment_estimator=tree, prediction_estimator=ols)
model.fit(X, y)
```
