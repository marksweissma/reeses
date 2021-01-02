


# @_predict_prediction_model.variant('predict_proba')
# def _predict_prediction_model(prediction_models, X, n_jobs):

    # # avoid storing the output of every estimator by summing them here
    # all_proba = [np.zeros((X.shape[0], j), dtype=np.float64)
                 # for j in np.atleast_1d(self.n_classes_)]

    # lock = threading.Lock()
    # Parallel(n_jobs=n_jobs, verbose=self.verbose,
             # **_joblib_parallel_args(require="sharedmem"))(
        # delayed(sk_forest._accumulate_prediction)(getattr(e.predict_proba, X, all_proba,
                                        # lock)
        # for e in self.estimators_)

    # for proba in all_proba:
        # proba /= len(self.estimators_)

    # if len(all_proba) == 1:
        # output = all_proba[0]
    # else:
        # output = all_proba
    # return output


    # leaf_predictions = {leaf: getattr(prediction_models[leaf], method)(np.vstack(_X)) for leaf, _X in Xs.items()}
    # predictions = _reconstruct(leaf_predictions, assignments)
    # return predictions


def predict(tree, reese, X, n_jobs):
    leaf_assignments = LeafAssignment.from_model(model, X)

    group_predictions = {}
    for group in leaf_assignments.leaves:
        model = reese.prediction_models_[group]
        data = leaf_assignments[group]
        group_predictions[group] = model.predict(data)

    predictions = leaf_assignments.reconstruct_from_groups(group_predictions)
    return predictions


@singledispatch
def predict(reese, X, method='predict'):


@predict.register(ClassifierMixin)
def _predict(reese, X, method='predict'):
    pass



@variants.primary
def predict_controller(reese, X, method, variant=None, ensemble_types=(BaseEnsemble,), **kwargs):

    tree_instance = reese.assignment_estimator
    prediction_instance = reese.prediction_estimator
    variant = get_variant(variant, tree_instance, prediction_instance, ensemble_types)

    predictions = getattr(predict_controller, variant)(reese, X, method, **kwargs)

    return predictions


@predict_controller.variant('single')
def predict_controller(reese, X, method, **kwargs):

    leaf_assignments = LeafAssignment.from_model(reese.assignment_estimator, X)


    predictions = predict_prediction_model(
            reese.assignment_estimator, reese.prediction_models_, reese, X, method=method
            )
    return predictions


@predict_controller.variant('ensemble')
def predict_controller(reese, X, method, **kwargs):
    predictions = [_parallel_predict_prediction_model(
        tree, prediction_models[idx], reese, X, method=method
        )
        for idx, tree in reese.prediction_models.items()]

    return ensemble_reducer(method, predictions)

