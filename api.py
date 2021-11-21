"""
Build API with FastAPI

Example from course content:
--------------------------------

# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union

from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

# Declare the data object with its components and their type.
class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list]
    item_id: int

app = FastAPI()

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/items/")
async def create_item(item: TaggedItem):
    return item


...

# A GET that in this case just returns the item_id we pass,
# but a future iteration may link the item_id here to the one we defined in our TaggedItem.
@app.get("/items/{item_id}")
async def get_items(item_id: int, count: int = 1):
    return {"fetch": f"Fetched {count} of {item_id}"}

# Note, parameters not declared in the path are automatically query parameters.

Path and query parameters are naturally strings since they are part of the endpoint URL. However, the type hints automatically convert the variables to their specified type. FastAPI automatically understands the distinction between path and query parameters by parsing the declaration. Note, to create optional query parameters use Optional from the typing module.

If we wanted to query the above API running on our local machine it would be via http://127.0.0.1:8000/items/42/?count=1.


Example with type hints
--------------------------------

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Value(BaseModel):
    value: int


@app.post("/{path}")
async def exercise_function(path: int, query: int, body: Value):
    return {"path": path, "query": query, "body": body}

"""

import pandas as pd
from pydantic import BaseModel
from pydantic import Field
from fastapi import FastAPI
from src.ml.model import load_model
from src.ml.data import load_config
import os


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc status")
    os.system("dvc remote list")
    os.system("dvc config -l")
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull -v") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# -------------------- GLOBAL VARIABLES --------------------
app = FastAPI()
config = load_config()
model = load_model(model_filepath=config["api"]["model_filepath"])


# Creating class to define the request body
# and the type hints of each attribute
class XTest(BaseModel):
    age: int
    capital_gain: float = Field(alias="capital-gain")
    capital_loss: float = Field(alias="capital-loss")
    education: str
    education_num: int = Field(alias="education-num")
    fnlgt: float
    hours_per_week: float = Field(alias="hours-per-week")
    marital_status: str = Field(alias="marital-status")
    native_country: str = Field(alias="native-country")
    occupation: str
    race: str
    relationship: str
    sex: str
    workclass: str

    class Config:
        schema_extra = {
            "example": {
                "age": 48,
                "workclass": "?",
                "fnlgt": 185291,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "?",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 6,
                "native-country": "United-States",
            }
        }


class YPredicted(BaseModel):
    class_label: str
    prediction: int


def parse_x_raw(data: XTest):
    """Load into dataframe and correct column names"""
    X_pred = pd.DataFrame(vars(data), index=[0])
    X_pred.columns = X_pred.columns.str.replace("_", "-")
    return X_pred


@app.get("/")
def welcome():
    return {"message": "Welcome! You called the GET method."}


@app.post("/predict")
def predict(data: XTest):
    X_pred = parse_x_raw(data)
    prediction = model.predict(X_pred)[0]
    class_label = config["data"]["target_labels"][prediction]

    return YPredicted(class_label=class_label, prediction=prediction)
