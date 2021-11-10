import pytest
import pandas as pd
import yaml
from src.ml.data import preprocess


@pytest.fixture(scope="session")
def df_census():
    return pd.read_csv("data/census.csv")


@pytest.fixture(scope="session")
def config():
    """Loads config.yaml from file"""
    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


@pytest.fixture(scope="session")
def X_sample_0(config):
    X, _ = preprocess(config)
    X_sample = X.sample(1, random_state=42).to_dict(orient="records")[0]
    return X_sample


@pytest.fixture(scope="session")
def X_sample_1(config):
    X, _ = preprocess(config)
    X_sample = X.sample(1, random_state=3).to_dict(orient="records")[0]
    return X_sample
