import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from utils import preprocess_text
from model import get_model
import json

MODEL_PATH = "finetune_model1.keras"
model = get_model(MODEL_PATH)

class ReqBody(BaseModel):
    text: str

INDEX_TO_CLASS = {
    0: 'Positive', 
    1: 'Neutral', 
    2: 'Negative'
}

def predict_sentiment(tokens):
    oup = model.predict(tokens, verbose=0)
    label = int(np.argmax(oup, axis=-1)[0])
    return {
        'sentiment': INDEX_TO_CLASS[label],
        'probs': oup[0].tolist()
    }

app = FastAPI()

@app.get("/")
def foo():
    return {
        "status": "Sentiment Classifier"
    }

@app.post("/predict")
def predict(req: ReqBody):
    text = req.text
    tokens = preprocess_text(text)

    result = predict_sentiment(tokens)

    return {
        'result': json.dumps(result)
    }