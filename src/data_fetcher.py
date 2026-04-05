import pandas as pd
import requests
import io

def fetch_jhu_data():
    """
    Fetches real-time COVID-19 data from JHU CSSE GitHub repository.
    Returns a cleaned DataFrame tailored for the project.
    """
    base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series"
    confirmed_url = f"{base_url}/time_series_covid19_confirmed_global.csv"
    deaths_url = f"{base_url}/time_series_covid19_deaths_global.csv"
    recovered_url = f"{base_url}/time_series_covid19_recovered_global.csv"

    def load_preprocess(url, value_name):
        try:
            r = requests.get(url)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            # Melt to long format
            df = df.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], 
                         var_name="Date", value_name=value_name)
            df['Date'] = pd.to_datetime(df['Date'])
            # Aggregating by Country/Region
            df = df.groupby(['Country/Region', 'Date'])[value_name].sum().reset_index()
            return df
        except Exception as e:
            print(f"Error fetching {value_name}: {e}")
            return None

    print("Fetching Confirmed Cases...")
    confirmed_df = load_preprocess(confirmed_url, "confirmed_cases")
    
    print("Fetching Deaths...")
    deaths_df = load_preprocess(deaths_url, "deaths")
    
    # Recovered data is no longer updated consistently in some JHU files, 
    # but we will try to fetch it or simulate it if missing.
    print("Fetching Recovered...")
    recovered_df = load_preprocess(recovered_url, "recovered")

    if confirmed_df is None:
        return None

    # Merge DataFrames
    df = confirmed_df.merge(deaths_df, on=['Country/Region', 'Date'], how='left')
    
    if recovered_df is not None:
        df = df.merge(recovered_df, on=['Country/Region', 'Date'], how='left')
    else:
        # Simple proxy for recovered if missing: (Confirmed 14 days ago) - Deaths
        df['recovered'] = df.groupby('Country/Region')['confirmed_cases'].shift(14).fillna(0) - df['deaths']
        df['recovered'] = df['recovered'].clip(lower=0)
    
    # Rename columns to match project standard
    df = df.rename(columns={'Country/Region': 'region', 'Date': 'date'})
    
    # Fill NA
    df = df.fillna(0)
    
    # Add dummy population (In real app, join with World Bank population data)
    # Here we just use a placeholder or heuristic
    df['population'] = 10_000_000 # Dummy value
    
    return df

if __name__ == "__main__":
    df = fetch_jhu_data()
    if df is not None:
        print(df.head())
        print(f"Regions: {df['region'].unique()[:5]}")
