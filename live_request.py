import requests
import json
import logging
from src.ml.data import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

request_data1 = {
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


request_data2 = {
    "age": 42,
    "workclass": "Private",
    "fnlgt": 159449,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec_managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 5178,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


if __name__ == "__main__":
    config = load_config()
    url = config["api"]["url"]

    response = requests.post(f"{url}/predict", data=json.dumps(request_data1))
    logging.info(f"Response code: {response.status_code}")
    logging.info(f"Response from API: {response.text}")

    response = requests.post(f"{url}/predict", data=json.dumps(request_data2))
    logging.info(f"Response code: {response.status_code}")
    logging.info(f"Response from API: {response.text}")
