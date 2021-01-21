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

