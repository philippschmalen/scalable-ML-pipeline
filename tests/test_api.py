"""
Tests capabilities of fastapi


Example
-------

from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
"""

from fastapi.testclient import TestClient
from api import app
from api import YPredicted
import json

# Instantiate the testing client with our app.
client = TestClient(app)


# Write tests using the same syntax as with the requests module.
def test_local_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert {"message": "Welcome! You called the GET method."} == json.loads(
        r.text
    )


def test_local_post_predict_class_0(X_sample_0):
    r = client.post("/predict", json=X_sample_0)
    assert r.status_code == 200
    assert YPredicted(class_label="<=50K", prediction=0) == YPredicted(
        **json.loads(r.text)
    )


def test_local_post_predict_class_1(X_sample_1):
    r = client.post("/predict", json=X_sample_1)
    assert r.status_code == 200
    assert YPredicted(class_label="<=50K", prediction=0) == YPredicted(
        **json.loads(r.text)
    )
