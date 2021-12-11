from src.ml.model import load_model
from src.ml.data import load_config
from src.ml.data import preprocess

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import (
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    false_positive_rate_difference,
    false_negative_rate_difference,
    equalized_odds_difference,
)


# Helper functions
def get_metrics_df(models_dict, y_true, group):
    """Returns dataframe for given model and group"""
    metrics_dict = {
        "Overall selection rate": (lambda x: selection_rate(y_true, x), True),
        "Demographic parity difference": (
            lambda x: demographic_parity_difference(
                y_true, x, sensitive_features=group
            ),
            True,
        ),
        "Demographic parity ratio": (
            lambda x: demographic_parity_ratio(
                y_true, x, sensitive_features=group
            ),
            True,
        ),
        "Overall balanced error rate": (
            lambda x: 1 - balanced_accuracy_score(y_true, x),
            True,
        ),
        "Balanced error rate difference": (
            lambda x: MetricFrame(
                metrics=balanced_accuracy_score,
                y_true=y_true,
                y_pred=x,
                sensitive_features=group,
            ).difference(method="between_groups"),
            True,
        ),
        "False positive rate difference": (
            lambda x: false_positive_rate_difference(
                y_true, x, sensitive_features=group
            ),
            True,
        ),
        "False negative rate difference": (
            lambda x: false_negative_rate_difference(
                y_true, x, sensitive_features=group
            ),
            True,
        ),
        "Equalized odds difference": (
            lambda x: equalized_odds_difference(
                y_true, x, sensitive_features=group
            ),
            True,
        ),
        "Overall AUC": (lambda x: roc_auc_score(y_true, x), False),
        "AUC difference": (
            lambda x: MetricFrame(
                metrics=roc_auc_score,
                y_true=y_true,
                y_pred=x,
                sensitive_features=group,
            ).difference(method="between_groups"),
            False,
        ),
    }

    df_dict = {}
    for metric_name, (metric_func, use_preds) in metrics_dict.items():

        df_dict[metric_name] = [
            metric_func(preds) for _, preds in models_dict.items()
        ]

    return pd.DataFrame.from_dict(
        df_dict, orient="index", columns=[group.name]
    )


if __name__ == "__main__":
    config = load_config()
    model = load_model()
    X, y = preprocess(config)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    y_pred = model.predict(X_test)
    roc_auc_score(y_test, y_pred)

    gm = MetricFrame(
        metrics=roc_auc_score,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=X_test["sex"],
    )

    sensitive_features = config["data"]["sensitive_features"]
    assert set(sensitive_features).issubset(
        set(X_test.columns)
    ), "At least one sensitive feature is not in X_test. Check selection in config."

    models_dict = {"LGBM": y_pred}

    pd.concat(
        [
            get_metrics_df(models_dict, y_test, X_test[group])
            for group in sensitive_features
        ],
        axis=1,
    ).to_csv("reports/fairness_metrics.csv", index=True)
