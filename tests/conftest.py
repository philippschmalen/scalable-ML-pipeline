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
def X_sample_0():
    return {
        "age": 31,
        "workclass": "State-gov",
        "fnlgt": 33308,
        "education": "Assoc-voc",
        "education-num": 11,
        "marital-status": "Married-civ-spouse",
        "occupation": "Craft-repair",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }


@pytest.fixture(scope="session")
def X_sample_1():
    return {
        "age": 46,
        "workclass": "Private",
        "fnlgt": 141483,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
