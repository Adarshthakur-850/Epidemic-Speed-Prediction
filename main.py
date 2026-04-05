import os
import pandas as pd
import numpy as np
from src.data_loader import load_data
from src.data_fetcher import fetch_jhu_data
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import feature_engineering_pipeline
from src.sir_model import SIRModel
from src.seir_model import SEIRModel
from src.forecasting import train_arima, train_lstm, predict_arima, predict_lstm
from src.evaluation import calculate_metrics
from src.visualization import plot_epidemic_curve, plot_growth_rate, plot_forecast

def main():
    print("=== Epidemic Speed Prediction Pipeline ===")
    
    # 1. Data Engineering
    print("\nStep 1: Loading Data...")
    print("Attempting to fetch live data from JHU...")
    try:
        df = fetch_jhu_data()
        if df is None:
            raise Exception("Fetch returned None")
        print("Successfully fetched live data.")
        # Filter for a specific region for demo purposes
        # Let's pick a country with good data, e.g., 'India', 'US', or 'Italy'
        # Or just pick the one with most cases
        top_region = df.groupby('region')['confirmed_cases'].max().idxmax()
        print(f"Selecting region with highest cases for demo: {top_region}")
        df = df[df['region'] == top_region]
    except Exception as e:
        print(f"Live data fetch failed ({e}). Generating synthetic data.")
        df = load_data() 

    print("\nStep 2: Preprocessing...")
    df = preprocess_pipeline(df)
    
    print("\nStep 3: Feature Engineering...")
    df = feature_engineering_pipeline(df)
    
    region = df['region'].iloc[0]
    print(f"Processing region: {region}")
    
    # 4. Visualization (EDA)
    print("\nStep 4: Visualizing Data...")
    plot_epidemic_curve(df, region)
    plot_growth_rate(df, region)
    
    # 5. Modeling
    population = df['population'].iloc[0]
    if population <= 0: population = 1_000_000
    
    # SIR
    print("\nStep 5a: SIR Modeling (Simple Reference)...")
    sir = SIRModel(N=population)
    sir.fit(df['confirmed_cases'], df['recovered']) # Placeholder fit
    
    # SEIR
    print("Step 5b: SEIR Modeling (Advanced with Optimization)...")
    seir = SEIRModel(N=population)
    # Estimate active infected
    infected_ts = (df['confirmed_cases'] - df['recovered'] - df['deaths']).clip(lower=0).values
    
    try:
        metrics = seir.fit(infected_ts)
        print(f"SEIR Optimization Results: {metrics}")
        print(f"Estimated Params: Beta={seir.beta:.4f}, Gamma={seir.gamma:.4f}, Sigma={seir.sigma:.4f}")
    except Exception as e:
        print(f"SEIR fitting failed: {e}")

    # 6. Forecasting Setup
    print("\nStep 6: Forecasting...")
    data_series = df[df['region'] == region].set_index('date')['new_confirmed']
    
    if len(data_series) > 50:
        # Split Train/Test (Last 30 days as test)
        test_size = 30
        train = data_series[:-test_size]
        test = data_series[-test_size:]
        
        # ARIMA
        print("- Training ARIMA...")
        try:
            arima_model = train_arima(train.values, order=(5,1,0))
            arima_pred = predict_arima(arima_model, steps=test_size)
        
            arima_metrics = calculate_metrics(test.values, arima_pred)
            print(f"ARIMA Metrics: {arima_metrics}")
            
            plot_forecast(train.index, train.values, test.index, test.values, arima_pred, 'ARIMA', region)
        except Exception as e:
            print(f"ARIMA failed: {e}")

        # LSTM for Growth Rate
        print("- Training LSTM...")
        # Using 'daily_growth_rate' for LSTM
        growth_series = df[df['region'] == region].set_index('date')['daily_growth_rate'].fillna(0)
        train_lstm_data = growth_series[:-test_size].values
        test_lstm_data = growth_series[-test_size:].values
        
        seq_length = 10
        if len(train_lstm_data) > seq_length:
            try:
                lstm_model, scaler = train_lstm(train_lstm_data, seq_length=seq_length, epochs=5) # Reduced epochs for demo speed
                
                # Predict
                last_seq_scaled = scaler.transform(train_lstm_data[-seq_length:].reshape(-1, 1))
                lstm_pred = predict_lstm(lstm_model, scaler, last_seq_scaled, steps=test_size)
                
                lstm_metrics = calculate_metrics(test_lstm_data, lstm_pred)
                print(f"LSTM Metrics: {lstm_metrics}")
                
                plot_forecast(train.index, train_lstm_data, test.index, test_lstm_data, lstm_pred, 'LSTM', region)
            except Exception as e:
                print(f"LSTM failed: {e}")
        else:
            print("Not enough data for LSTM.")
    else:
        print("Dataset too small for train/test split forecasting.")
    
    print("\n=== Pipeline Complete ===")
    print("Outputs saved in 'outputs/'")
    print("\nTo run the Real-Time Dashboard:")
    print("  streamlit run dashboard.py")
    print("\nTo run the API:")
    print("  uvicorn api:app --reload")

if __name__ == "__main__":
    main()
