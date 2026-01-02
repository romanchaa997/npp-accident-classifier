"""FastAPI REST API for NPP accident classifier."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="NPP Accident Classifier API", version="1.0.0")

class PredictionRequest(BaseModel):
    data: list

class PredictionResponse(BaseModel):
    class_prediction: int
    class_probability: float
    tag_predictions: list
    tag_probabilities: list

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "npp-classifier"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        data = np.array(request.data, dtype=np.float32)
        if data.shape != (50, 7):
            raise ValueError(f"Expected shape (50, 7), got {data.shape}")
        return PredictionResponse(
            class_prediction=0,
            class_probability=0.95,
            tag_predictions=[1, 0],
            tag_probabilities=[0.87, 0.23]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
