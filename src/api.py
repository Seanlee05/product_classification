# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, constr
import joblib
import os
import sys

# Ensure the root directory is in sys.path so imports work during tests
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import preprocess_text

# Define absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "product_classifier.pkl")

# Load model GLOBALLY and IMMEDIATELY
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"SUCCESS: Model loaded from {MODEL_PATH}")
    else:
        model = None
        print(f"ERROR: Model file not found at {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"ERROR: Could not load model: {e}")

class ProductRequest(BaseModel):
    description: constr(strip_whitespace=True, min_length=2) 

class ProductResponse(BaseModel):
    description: str
    predicted_category: str

app = FastAPI(title="Product Classification API")

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact missing.")
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict", response_model=ProductResponse)
def predict_category(request: ProductRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
    
    try:
        cleaned_text = preprocess_text(request.description)
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Invalid input description.")
            
        prediction = model.predict([cleaned_text])[0]
        return ProductResponse(
            description=request.description,
            predicted_category=prediction
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))