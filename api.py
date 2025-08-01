from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
from typing import List

app = FastAPI()

#Fixing CORS headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://smart-gear-1.onrender.com/"],  
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Authorization", "Content-Type"],
)

model = joblib.load("model.pkl")

#Specifes the data type of the input
class PredictionInput(BaseModel):
    values: List[int]

#The api endpoint to get predictions from the Isolation Forest model
@app.post("/predict")
def get_prediction(input_data: PredictionInput):
    prediction = model.predict([input_data.values])
    return {"Preditction": prediction.tolist()}