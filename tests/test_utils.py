import pytest
import numpy as np
from numpy import testing as npt
from reeses.utils import (get_shapes, package_arrays, GroupAssignment,
                          MeanClassifier, MeanRegressor)

SHAPES =[({0: None}, {}),
         ({0: []}, {}),
         ({0: np.array([1])}, {0: (1,)}),
         ({0: np.array([1, 1])}, {0: (2,)}),
         ({0: np.array([[1], [1]])}, {0: (2, 1)}),
         ({0: np.array([[1, 1]])}, {0: (1, 2)}),
         ({0: np.array([1, 1]), 1: np.arange(4).reshape(2, 2)}, {0: (2,), 1: (2, 2)})
         ]

@pytest.mark.parametrize("dict_arr,shape", SHAPES)
def test_get_shapes(dict_arr, shape):
    assert get_shapes(dict_arr) ==  shape

PACKAGES = [([np.array([1]), np.array([1])], (0,), np.array([1, 1])),
            ([np.array([1]), np.array([1])], (1,), np.array([1, 1])),
            ([np.array([1]), np.array([1])], (1, 1), np.array([[1], [1]])),
            ([np.array([1, 1]), np.array([1, 1])], (1, 2), np.ones((2, 2))),
            ]

@pytest.mark.parametrize("arrs,shape,arr", PACKAGES)
def test_package_arrays(arrs, shape, arr):
    npt.assert_array_almost_equal(package_arrays(arrs, shape), arr)


def assert_array_dict_almost_equal(a, b):
    assert sorted(a) == sorted(b)
    for key in a:
        npt.assert_array_almost_equal(a[key], b[key])


GROUPS_1 = [0, 1, 2, 3, 4]
GROUPS_2 = [2, 2, 1, 1, 1]
X = np.arange(10).reshape(5, 2)
WEIGHT_1 = np.arange(5)
WEIGHT_2 = np.array([2, 2, 2, 1, 1])

PACKAGE_1 = {'X': {idx: [row] for idx, row in enumerate(X)},
             'sample_weight': {idx: [value] for idx, value in enumerate(WEIGHT_1)}
             }

PACKAGE_2 = {'X': {2: list(np.arange(4).reshape(2, 2)), 1: list(np.arange(6).reshape(3, 2) + 4)},
             'sample_weight': {2: np.array([2, 2]), 1: np.array([2, 1, 1])}
             }

GROUPS = [(GROUPS_1, X, WEIGHT_1, PACKAGE_1),
          (GROUPS_2, X, WEIGHT_2, PACKAGE_2),
          ]
@pytest.mark.parametrize("groups,X,sample_weight,packaged", GROUPS)
def test_group_assignments(groups, X, sample_weight, packaged):
    group_assignment = GroupAssignment.from_groups(groups, X=X,sample_weight=sample_weight)
    for key in ['X', 'sample_weight']:
        assert_array_dict_almost_equal(group_assignment[key], packaged[key])




