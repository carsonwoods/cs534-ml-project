"""
Generic Model Tuning
"""

# 3rd party imports
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    fbeta_score,
    make_scorer,
)
from sklearn.model_selection import (
    GridSearchCV,
    ShuffleSplit,
)


# Model Thunderdome(?) (many will enter, one will leave)
def build_generic_model(
    model,
    params,
    train_x,
    train_y,
    test_x,
    test_y,
    num_splits=5,
    refit_str="AUC",
    verbose_lvl=10,
):
    """
    Performs Monte-Carlo Cross Validation on a generic module.
    """
    results = {}
    f_score_mode = "weighted"

    scoring_metrics = {}
    scoring_metrics["AUC"] = make_scorer(
        roc_auc_score, multi_class="ovr", needs_proba=True
    )
    scoring_metrics["ACC"] = "accuracy"
    scoring_metrics["F1"] = make_scorer(f1_score, average=f_score_mode)
    scoring_metrics["F2"] = make_scorer(
        fbeta_score, beta=2, average=f_score_mode
    )

    # instantiate GridSearchCV with MCCV
    grid_search_model = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring=scoring_metrics,
        refit=refit_str,
        cv=ShuffleSplit(n_splits=num_splits, test_size=0.20),
        verbose=verbose_lvl,
        n_jobs=-1,
        return_train_score=True,
    )

    grid_search_model.fit(train_x, train_y)

    # return best estimator for evaluating test score
    results["best-model"] = grid_search_model.best_estimator_
    results["fit_time"] = grid_search_model.cv_results_["mean_fit_time"][
        grid_search_model.best_index_
    ]
    results["train-auc"] = grid_search_model.cv_results_["mean_train_AUC"][
        grid_search_model.best_index_
    ]
    results["train-acc"] = grid_search_model.cv_results_["mean_train_ACC"][
        grid_search_model.best_index_
    ]
    results["train-f1"] = grid_search_model.cv_results_["mean_train_F1"][
        grid_search_model.best_index_
    ]
    results["train-f2"] = grid_search_model.cv_results_["mean_train_F2"][
        grid_search_model.best_index_
    ]
    results["val-auc"] = grid_search_model.cv_results_["mean_test_AUC"][
        grid_search_model.best_index_
    ]
    results["val-acc"] = grid_search_model.cv_results_["mean_test_ACC"][
        grid_search_model.best_index_
    ]
    results["val-f1"] = grid_search_model.cv_results_["mean_test_F1"][
        grid_search_model.best_index_
    ]
    results["val-f2"] = grid_search_model.cv_results_["mean_test_F2"][
        grid_search_model.best_index_
    ]

    results["test-auc"] = roc_auc_score(
        test_y,
        results["best-model"].predict_proba(test_x),
        multi_class="ovr",
    )
    results["test-acc"] = accuracy_score(
        test_y, results["best-model"].predict(test_x)
    )
    results["test-f1"] = f1_score(
        test_y,
        results["best-model"].predict(test_x),
        average=f_score_mode,
    )
    results["test-f2"] = fbeta_score(
        test_y,
        results["best-model"].predict(test_x),
        beta=2,
        average=f_score_mode,
    )

    return results
