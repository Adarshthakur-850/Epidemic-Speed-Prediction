import pandas as pd
import numpy as np

def add_daily_features(df):
    """
    Calculates daily new cases from cumulative data.
    """
    # Assuming data is sorted by date within region
    df['new_confirmed'] = df.groupby('region')['confirmed_cases'].diff().fillna(0)
    df['new_deaths'] = df.groupby('region')['deaths'].diff().fillna(0)
    df['new_recovered'] = df.groupby('region')['recovered'].diff().fillna(0)
    
    # Handle negative values which can happen due to data corrections
    df['new_confirmed'] = df['new_confirmed'].clip(lower=0)
    df['new_deaths'] = df['new_deaths'].clip(lower=0)
    df['new_recovered'] = df['new_recovered'].clip(lower=0)
    
    return df

def add_growth_features(df):
    """
    Calculates growth rates and doubling times.
    """
    # Growth factor: new cases today / new cases yesterday
    df['growth_factor'] = df.groupby('region')['new_confirmed'].pct_change().add(1).fillna(1)
    df['growth_factor'] = df['growth_factor'].replace([np.inf, -np.inf], 1)
    
    # Daily growth rate (percentage)
    df['daily_growth_rate'] = df.groupby('region')['confirmed_cases'].pct_change().fillna(0)
    df['daily_growth_rate'] = df['daily_growth_rate'].replace([np.inf, -np.inf], 0)
    
    return df

def add_rolling_features(df, window=7):
    """
    Adds rolling averages to smooth out weekly seasonality.
    """
    df[f'new_confirmed_roll{window}'] = df.groupby('region')['new_confirmed'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df[f'growth_factor_roll{window}'] = df.groupby('region')['daily_growth_rate'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    return df

def add_lag_features(df, lags=[1, 7, 14]):
    """
    Adds lagged features for forecasting.
    """
    for lag in lags:
        df[f'new_confirmed_lag{lag}'] = df.groupby('region')['new_confirmed'].shift(lag).fillna(0)
    return df

def feature_engineering_pipeline(df):
    """
    Runs full feature engineering pipeline.
    """
    df = add_daily_features(df)
    df = add_growth_features(df)
    df = add_rolling_features(df)
    df = add_lag_features(df)
    return df

if __name__ == "__main__":
    # Test feature engineering
    from data_loader import load_data
    from preprocessing import preprocess_pipeline
    
    df = load_data()
    df = preprocess_pipeline(df)
    df = feature_engineering_pipeline(df)
    print(df[['date', 'new_confirmed', 'growth_factor', 'new_confirmed_roll7']].head(10))
