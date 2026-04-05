import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(days=365, start_date='2023-01-01', region='Region_A', population=1000000):
    """
    Generates a synthetic epidemic dataset with SIR-like dynamics.
    """
    date_range = pd.date_range(start=start_date, periods=days, freq='D')
    
    # Simulate SIR dynamics
    t = np.linspace(0, 50, days)
    # Simple logistic-like growth for infected, then decay
    confirmed = population * (1 / (1 + np.exp(-(t - 10))))  # Sigmoid curve
    # Add some noise
    confirmed += np.random.normal(0, population * 0.001, days)
    confirmed = np.clip(confirmed, 0, population)
    
    # Recovered lags behind confirmed
    recovered = confirmed * 0.8
    # Deaths are a fraction of confirmed
    deaths = confirmed * 0.02
    
    df = pd.DataFrame({
        'date': date_range,
        'region': region,
        'confirmed_cases': confirmed.astype(int),
        'deaths': deaths.astype(int),
        'recovered': recovered.astype(int),
        'population': population
    })
    
    # Ensure confirmed is strictly non-decreasing (cumulative)
    df['confirmed_cases'] = df['confirmed_cases'].sort_values().values
    df['deaths'] = df['deaths'].sort_values().values
    df['recovered'] = df['recovered'].sort_values().values

    return df

def load_data(filepath=None):
    """
    Loads dataset from a CSV file or generates synthetic data if filepath is None.
    """
    if filepath:
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded data from {filepath}")
            return df
        except FileNotFoundError:
            print(f"File {filepath} not found. Generating synthetic data.")
            return generate_synthetic_data()
    else:
        print("No filepath provided. Generating synthetic data.")
        return generate_synthetic_data()

if __name__ == "__main__":
    # Test data loader
    df = load_data()
    print(df.head())
