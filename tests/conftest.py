import pytest
import pandas as pd
import yaml


@pytest.fixture(scope="session")
def df_census():
    return pd.read_csv("data/census.csv")


@pytest.fixture(scope="session")
def config():
    """Loads config.yaml from file"""
    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg
