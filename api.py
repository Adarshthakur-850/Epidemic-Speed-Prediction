from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from src.data_fetcher import fetch_jhu_data
from src.preprocessing import preprocess_pipeline
from src.seir_model import SEIRModel

app = FastAPI(title="Epidemic Speed Prediction API", version="1.0")

# In-memory cache for data
DATA_CACHE = None

def get_data():
    global DATA_CACHE
    if DATA_CACHE is None:
        print("Fetching live data...")
        try:
            df = fetch_jhu_data()
            if df is not None:
                DATA_CACHE = preprocess_pipeline(df)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    return DATA_CACHE

class PredictionRequest(BaseModel):
    region: str
    days: int

@app.get("/")
def home():
    return {"message": "Epidemic Speed Prediction API is running."}

@app.get("/regions")
def get_regions():
    df = get_data()
    if df is None:
        raise HTTPException(status_code=503, detail="Data unavailable")
    regions = df['region'].unique().tolist()
    return {"regions": regions}

@app.get("/data/{region}")
def get_region_data(region: str):
    df = get_data()
    if df is None:
        raise HTTPException(status_code=503, detail="Data unavailable")
    
    region_df = df[df['region'] == region]
    if region_df.empty:
        raise HTTPException(status_code=404, detail="Region not found")
        
    return region_df.to_dict(orient="records")

@app.post("/predict/seir")
def predict_seir(request: PredictionRequest):
    df = get_data()
    if df is None:
        raise HTTPException(status_code=503, detail="Data unavailable")
    
    region = request.region
    region_df = df[df['region'] == region]
    
    if region_df.empty:
        raise HTTPException(status_code=404, detail="Region not found")
    
    # Fit SEIR
    population = region_df['population'].iloc[0]
    # If population is dummy/0, use a default to avoid crash, though fit might be poor
    if population <= 0: population = 1_000_000
        
    model = SEIRModel(N=population)
    
    # Use 'confirmed_cases' - 'recovered' - 'deaths' as approx for Active Infected
    # This is a simplification.
    infected = (region_df['confirmed_cases'] - region_df['deaths'] - region_df['recovered']).clip(lower=0).values
    
    metrics = model.fit(infected)
    
    # Predict
    future_days = request.days
    current_I = infected[-1] if len(infected) > 0 else 0
    forecast_df = model.predict(days=future_days + 30, I0=current_I) # Predict slightly more for context
    
    return {
        "region": region,
        "metrics": metrics,
        "params": {
            "beta": model.beta,
            "gamma": model.gamma,
            "sigma": model.sigma,
            "R0": metrics.get('R0_est')
        },
        "forecast": forecast_df.to_dict(orient="records")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
