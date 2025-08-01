from fastapi import FastAPI, WebSocket
# from app.fraud_detection import fraud_detector  # Import model from ml_model.py
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")

class PredictionInput(BaseModel):
    values: List[int]

@app.get("/")
def home():
    return {"message": "FastAPI is running!"}

@app.post("/predict")

def get_prediction(input_data: PredictionInput):
    prediction = model.predict([input_data.values])
    return {"Preditction": prediction.tolist()}