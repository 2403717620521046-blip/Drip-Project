from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Financial Distress Prediction API",
    description="Machine Learning API for Financial Distress Prediction",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "classification_model.joblib"
)

try:

    model = joblib.load(MODEL_PATH)

    model_loaded = True

    print("Model loaded successfully.")

except Exception as e:

    model = None

    model_loaded = False

    print("Error loading model:")
    print(e)


# ============================================================
# REQUEST DATA MODEL
# ============================================================

class PredictionRequest(BaseModel):

    x1: float | None = None
    x2: float | None = None
    x3: float | None = None
    x4: float | None = None
    x5: float | None = None
    x6: float | None = None
    x7: float | None = None
    x8: float | None = None
    x9: float | None = None
    x10: float | None = None
    x11: float | None = None
    x12: float | None = None
    x13: float | None = None
    x14: float | None = None
    x15: float | None = None
    x16: float | None = None
    x17: float | None = None
    x18: float | None = None
    x19: float | None = None
    x20: float | None = None
    x21: float | None = None
    x22: float | None = None
    x23: float | None = None
    x24: float | None = None
    x25: float | None = None
    x26: float | None = None
    x27: float | None = None
    x28: float | None = None
    x29: float | None = None
    x30: float | None = None
    x31: float | None = None
    x32: float | None = None
    x33: float | None = None
    x34: float | None = None
    x35: float | None = None
    x36: float | None = None
    x37: float | None = None
    x38: float | None = None
    x39: float | None = None
    x40: float | None = None
    x41: float | None = None
    x42: float | None = None
    x43: float | None = None
    x44: float | None = None
    x45: float | None = None
    x46: float | None = None
    x47: float | None = None
    x48: float | None = None
    x49: float | None = None
    x50: float | None = None
    x51: float | None = None
    x52: float | None = None
    x53: float | None = None
    x54: float | None = None
    x55: float | None = None
    x56: float | None = None
    x57: float | None = None
    x58: float | None = None
    x59: float | None = None
    x60: float | None = None
    x61: float | None = None
    x62: float | None = None
    x63: float | None = None
    x64: float | None = None
    x65: float | None = None
    x66: float | None = None
    x67: float | None = None
    x68: float | None = None
    x69: float | None = None
    x70: float | None = None
    x71: float | None = None
    x72: float | None = None
    x73: float | None = None
    x74: float | None = None
    x75: float | None = None
    x76: float | None = None
    x77: float | None = None
    x78: float | None = None
    x79: float | None = None

    # x80 is categorical in the dataset
    x80: str | None = None

    x81: float | None = None
    x82: float | None = None
    x83: float | None = None


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Financial Distress Prediction API",
        "status": "running",
        "model_loaded": model_loaded,
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    if model_loaded:

        return {
            "status": "healthy",
            "model_loaded": True
        }

    return {
        "status": "unhealthy",
        "model_loaded": False
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model": "Financial Distress Classification Model",
        "target": "Distress_Class",
        "class_0": "Not Financially Distressed",
        "class_1": "Financially Distressed",
        "input_features": 83
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    if not model_loaded:

        raise HTTPException(
            status_code=500,
            detail="Machine learning model could not be loaded."
        )

    try:

        # Convert request into dictionary
        input_data = request.model_dump()

        # Remove values that were not supplied
        input_data = {
            key: value
            for key, value in input_data.items()
            if value is not None
        }

        # Convert dictionary to DataFrame
        input_df = pd.DataFrame(
            [input_data]
        )

        # Make prediction
        prediction = model.predict(
            input_df
        )[0]

        # Default probability
        probability = None

        # Get probability if supported
        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                input_df
            )[0]

            probability = float(
                probabilities[1]
            )

        # Convert result
        if int(prediction) == 1:

            result = "Financially Distressed"

        else:

            result = "Not Financially Distressed"

        return {
            "success": True,
            "prediction": int(prediction),
            "result": result,
            "distress_probability": probability
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
