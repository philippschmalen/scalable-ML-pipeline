"""
Main executable to load data, train model, and evaluate the model.
"""
from ml.data import preprocess
from ml.data import load_config
from ml.model import train_model
from ml.model import compute_model_metrics
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="logs/train_model.log",
)


def go():
    "Model training pipeline"

    config = load_config()
    X, y = preprocess(config)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # model training and export
    model = train_model(X_train, y_train, config, export_pipeline=True)
    y_pred = model.predict(X_test)
    compute_model_metrics(y_test, y_pred)


if __name__ == "__main__":
    go()
