import pandas as pd

def clean_data(df):
    """
    Cleans the dataset:
    - Parses dates
    - Handles missing values
    - Sorts by date
    """
    # Parse dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values(by=['region', 'date'])
    
    # Handle missing values (forward fill for cumulative data)
    df = df.fillna(method='ffill').fillna(0)
    
    return df

def normalize_data(df):
    """
    Normalizes cases, deaths, and recovered by population.
    Adds columns: confirmed_norm, deaths_norm, recovered_norm
    """
    if 'population' in df.columns:
        df['confirmed_norm'] = df['confirmed_cases'] / df['population']
        df['deaths_norm'] = df['deaths'] / df['population']
        df['recovered_norm'] = df['recovered'] / df['population']
    
    return df

def preprocess_pipeline(df):
    """
    Runs the full preprocessing pipeline.
    """
    df = clean_data(df)
    df = normalize_data(df)
    return df

if __name__ == "__main__":
    # Test preprocessing
    from data_loader import load_data
    df = load_data()
    df_processed = preprocess_pipeline(df)
    print(df_processed.head())
