from pydantic import BaseModel
from datetime import datetime

class PredictionInput(BaseModel):
    datetime: datetime
    season: int
    holiday: int
    workingday: int
    weather: int
    temp: float
    atemp: float
    humidity: float
    windspeed: float

class PredictionOutput(BaseModel):
    model_name: str
    model_version: str
    prediction: float
    timestamp: datetime
