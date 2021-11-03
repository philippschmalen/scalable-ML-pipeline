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

from pydantic import BaseModel
from pydantic import Field
from fastapi import FastAPI
from src.ml.model import load_model


app = FastAPI()
model = load_model()


# Creating class to define the request body
# and the type hints of each attribute
class body(BaseModel):
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


# sample data
# {'age': 41,
#  'capital-gain': 0,
#  'capital-loss': 0,
#  'education': '12th',
#  'education-num': 8,
#  'fnlgt': 327606,
#  'hours-per-week': 40,
#  'marital-status': 'Separated',
#  'native-country': 'United-States',
#  'occupation': 'Craft-repair',
#  'race': 'Black',
#  'relationship': 'Not-in-family',
#  'sex': 'Male',
#  'workclass': 'Private'}


@app.get("/")
def welcome():
    return {"message": "Welcome! You called the GET method."}


@app.post("/predict")
def predict(body):
    return {"message": "Welcome! You called the POST method."}
