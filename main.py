from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import yfinance as yf
import requests
import os

app = FastAPI()

# ===== CORS (ALLOW FRONTEND ON VERCEL) =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # open for testing, restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MODEL SETUP =====
MODEL_URL = "https://github.com/Harshgoyal2004/stocksight_backend/raw/refs/heads/main/lstm_model.h5"
MODEL_PATH = "lstm_model.h5"

# download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

model = tf.keras.models.load_model(MODEL_PATH)


# ===== DATA PREPARATION FUNCTION =====
def get_last_100(stock):
    df = yf.download(stock, period="1y")
    close_prices = df["Close"].values

    if len(close_prices) < 100:
        raise ValueError("Not enough data for stock")

    last100 = close_prices[-100:]
    return last100


# ===== PREDICTION (30 days ahead) =====
def predict_next_30(model, last100):
    seq = last100.reshape(1, 100, 1)
    predictions = []

    for _ in range(30):
        pred = model.predict(seq)[0][0]
        predictions.append(float(pred))

        seq = np.append(seq[:, 1:, :], [[[pred]]], axis=1)

    return predictions


# ===== API ENDPOINT (MATCHES YOUR FRONTEND EXACTLY) =====
@app.post("/predict")
async def predict(payload: dict):
    stock = payload["stock"].upper()

    try:
        last100 = get_last_100(stock)
        predictions = predict_next_30(model, last100)

        return {
            "last100": last100.tolist(),
            "predictions": predictions
        }

    except Exception as e:
        return {"error": str(e)}
