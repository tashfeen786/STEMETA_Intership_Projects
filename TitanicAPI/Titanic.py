from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the trained ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Pydantic model for input data
class Passenger(BaseModel):
    Pclass: int
    Sex: int      # 0 = female, 1 = male
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int # 0 = C, 1 = Q, 2 = S

# Root route
@app.get("/")
def read_root():
    return {"message": "Titanic Survival Prediction API is Live!"}

# Prediction route
@app.post("/predict")
def predict_survival(passenger: Passenger):
    data = [[
        passenger.Pclass,
        passenger.Sex,
        passenger.Age,
        passenger.SibSp,
        passenger.Parch,
        passenger.Fare,
        passenger.Embarked
    ]]
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
# To run the FastAPI app, use the command: