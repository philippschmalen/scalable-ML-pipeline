"""
Test model capabilities
"""

from flaml import AutoML
from src.ml.data import load_config
from src.ml.model import create_automl_pipeline


def test_config_automl_exist(config):
    assert (
        config.get("automl") is not None
    ), "automl settings not found in config"


def test_config_automl_params_exist(config):
    """Checks that all params from config are expected"""
    config = load_config().get("automl")
    params = config.keys()
    params_expected = [
        "task",
        "time_budget",
        "random_state",
        "metric",
        "log_training_metric",
        "log_file_name",
        "eval_method",
        "estimator_list",
    ]

    assert set(params).issubset(
        params_expected
    ), "Some parameters for automl in config.yaml are not expected. See https://microsoft.github.io/FLAML/ for which ones are allowed"


def test_pipeline_automl_type(config):
    """Checks that pipeline estimator has expected type"""
    assert isinstance(
        create_automl_pipeline(config).steps[-1][-1], AutoML
    ), "Estimator in pipeline is not of expected type flaml.automl.AutoML"
