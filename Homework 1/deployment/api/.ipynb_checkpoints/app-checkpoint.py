from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class FruitFeatures(BaseModel):
    mass: float
    width: float
    height: float
    color_score: float

@app.post("/predict")
async def predict(features: FruitFeatures):
    model = joblib.load("models/fruit_classifier.joblib")
    prediction = model.predict([[
        features.mass,
        features.width,
        features.height,
        features.color_score
    ]])
    return {"prediction": int(prediction[0])}