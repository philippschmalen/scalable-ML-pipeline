"""
Create sklearn pipeline, run automl, store model and compute metrics
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from flaml import AutoML
import joblib
import logging
from datetime import datetime
import glob
import os


def get_latest_file(filepath):
    return max(glob.iglob(filepath), key=os.path.getctime)


def create_automl_pipeline(config):
    """Build automl pipeline with numeric and categorical features"""

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                categorical_transformer,
                list(config["data"]["cat_features"]),
            ),
            ("num", numeric_transformer, list(config["data"]["num_features"])),
        ]
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("automl", AutoML())]
    )

    return pipeline


def train_model(X_train, y_train, config, export_pipeline=True):

    timestamp = datetime.utcnow().strftime("%y%m%d_%H%M%S")
    pipeline = create_automl_pipeline(config)

    settings_automl = {
        "automl__task": config["automl"]["task"],
        "automl__metric": config["automl"]["metric"],
        "automl__log_training_metric": config["automl"]["log_training_metric"],
        "automl__log_file_name": config["automl"]["log_file_name"],
        "automl__time_budget": config["automl"]["time_budget"],
        "automl__seed": config["automl"]["random_state"],
        "automl__estimator_list": config["automl"]["estimator_list"],
    }

    logging.info(f"Train model with settings: {settings_automl}")
    pipeline.fit(X_train, y_train, **settings_automl)
    logging.info("Model training finished")

    if export_pipeline:

        automl_best_estimator = pipeline.steps[-1][-1]
        filepath = f"model/automl_pipeline_{timestamp}_{automl_best_estimator.best_estimator}.pkl"
        joblib.dump(pipeline, filepath)
        logging.info(f"Saving prediction pipeline to {filepath}.")

    return pipeline


def compute_model_metrics(y, preds, export_metrics=True):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    auc = roc_auc_score(y, preds, average="weighted")

    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1: {fbeta}")
    logging.info(f"AUC: {auc}")

    if export_metrics:
        timestamp = datetime.utcnow().strftime("%y%m%d_%H%M%S")
        with open(f"reports/metrics_{timestamp}.txt", "w") as f:
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1: {fbeta}\n")
            f.write(f"AUC: {auc}\n")

    return precision, recall, fbeta, auc


def load_model(model_filepath="model/*.pkl"):
    "Loads latest model from model/*.pkl"
    model_pkl = get_latest_file(model_filepath)
    logging.info(f"Loaded model from {model_pkl}")
    return joblib.load(model_pkl)
