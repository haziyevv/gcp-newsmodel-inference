import os
import yaml
import uvicorn
from pydantic import BaseModel
from typing import List

from fastapi import Request, FastAPI, Response
from .inference import Inference

# read the config file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


app = FastAPI()
AIP_HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
AIP_PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")

inference = Inference(
    model_name=config["model_name"],
    tokenizer_name=config["tokenizer_name"],
    max_length=config["max_seq_length"],
)


class Prediction(BaseModel):
    category: str


class Predictions(BaseModel):
    predictions: List[Prediction]


@app.get(AIP_HEALTH_ROUTE, status_code=200)
async def health():
    return {"health": "ok"}


@app.post(AIP_PREDICT_ROUTE, response_model=Predictions)
async def predict(request: Request):
    body = await request.json()
    instances = body["instances"]

    pred_cats = inference.predict(instances)

    predicted_categories = Predictions(
        predictions=[
            Prediction(category=inference.index2cat[x.item()]) for x in pred_cats
        ]
    )

    return predicted_categories


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
    )
