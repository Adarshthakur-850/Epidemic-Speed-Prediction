import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from src.data_fetcher import fetch_jhu_data
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import feature_engineering_pipeline
from src.seir_model import SEIRModel
from src.forecasting import train_arima, predict_arima

# Page Config
st.set_page_config(page_title="Epidemic Dashboard", layout="wide")

st.title("🦠 Real-Time Epidemic Speed Prediction Dashboard")
st.markdown("Monitor COVID-19 spread speed, visualize trends, and run advanced forecasting models.")

# Sidebar
st.sidebar.header("Settings")
data_source = st.sidebar.radio("Data Source", ["Live JHU Data", "Synthetic Data"])

@st.cache_data
def load_and_prep_data(source):
    if source == "Live JHU Data":
        with st.spinner("Fetching live data from JHU..."):
            df = fetch_jhu_data()
            if df is None:
                st.error("Failed to fetch live data. Falling back to synthetic.")
                from src.data_loader import generate_synthetic_data
                df = generate_synthetic_data()
    else:
        from src.data_loader import generate_synthetic_data
        df = generate_synthetic_data()
    
    df = preprocess_pipeline(df)
    df = feature_engineering_pipeline(df)
    return df

try:
    df = load_and_prep_data(data_source)
    
    # Region Selection
    regions = df['region'].unique()
    selected_region = st.sidebar.selectbox("Select Region", regions)
    
    region_df = df[df['region'] == selected_region]
    
    # KPIs
    latest = region_df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Confirmed Cases", f"{latest['confirmed_cases']:,.0f}")
    col2.metric("Deaths", f"{latest['deaths']:,.0f}")
    col3.metric("Daily Growth Rate", f"{latest['daily_growth_rate']:.2%}")
    col4.metric("Growth Factor", f"{latest['growth_factor']:.2f}")
    
    # Charts
    st.subheader("Epidemic Curves")
    fig_curve = px.line(region_df, x='date', y=['confirmed_cases', 'deaths', 'recovered'], 
                        title=f"Epidemic Trajectory - {selected_region}")
    st.plotly_chart(fig_curve, use_container_width=True)
    
    st.subheader("Daily Growth Rate using Rolling Average")
    # Using 7-day rolling for smoother visualization
    fig_growth = px.line(region_df, x='date', y='daily_growth_rate', 
                         title=f"Daily Growth Rate Trend - {selected_region}")
    st.plotly_chart(fig_growth, use_container_width=True)
    
    # Modeling Section
    st.markdown("---")
    st.header("Advanced Modeling & Forecasting")
    
    model_choice = st.selectbox("Choose Model", ["ARIMA (Time-Series)", "SEIR (Compartmental)"])
    
    if model_choice == "ARIMA (Time-Series)":
        steps = st.slider("Forecast Days", 7, 60, 30)
        if st.button("Run ARIMA Forecast"):
            with st.spinner("Training ARIMA..."):
                train_data = region_df['new_confirmed'].values
                # Simple train on all data for forecast
                model_fit = train_arima(train_data)
                forecast = model_fit.forecast(steps=steps)
                
                # Plot
                last_date = region_df['date'].max()
                future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps)
                
                fig_arima = go.Figure()
                fig_arima.add_trace(go.Scatter(x=region_df['date'], y=train_data, name='Historical New Cases'))
                fig_arima.add_trace(go.Scatter(x=future_dates, y=forecast, name='ARIMA Forecast', line=dict(color='red', dash='dash')))
                fig_arima.update_layout(title=f"ARIMA Forecast for {selected_region}")
                st.plotly_chart(fig_arima, use_container_width=True)
                
    elif model_choice == "SEIR (Compartmental)":
        st.info("SEIR estimates parameters (Beta, Gamma, Sigma) automatically from the data.")
        fit_days = st.slider("Days to fit parameters on (from start)", 30, len(region_df), len(region_df))
        forecast_days = st.slider("Forecast Horizon", 30, 180, 60)
        
        if st.button("Run SEIR Simulation"):
            with st.spinner("Optimizing SEIR Parameters..."):
                population = region_df['population'].iloc[0]
                if population == 0: population = 1_000_000 # Fallback
                
                seir = SEIRModel(N=population)
                # Proxy for Active Infected
                infected_data = (region_df['confirmed_cases'] - region_df['recovered'] - region_df['deaths']).values
                infected_data = np.clip(infected_data, 0, None)
                
                metrics = seir.fit(infected_data, days_to_fit=fit_days)
                
                st.write(f"**Estimated R0**: {metrics['R0_est']:.2f}")
                st.write(f"**Parameters**: Beta={seir.beta:.4f}, Gamma={seir.gamma:.4f}, Sigma={seir.sigma:.4f}")
                
                # Predict
                predictions = seir.predict(days=fit_days + forecast_days, I0=infected_data[0])
                
                # Plot
                fig_seir = go.Figure()
                # Actual
                fig_seir.add_trace(go.Scatter(x=region_df['date'][:len(infected_data)], y=infected_data, name='Actual Infected', mode='markers'))
                # Model
                # Create dates for prediction
                start_date = region_df['date'].min()
                pred_dates = [start_date + pd.Timedelta(days=int(i)) for i in predictions['day']]
                
                fig_seir.add_trace(go.Scatter(x=pred_dates, y=predictions['Infected'], name='Model Infected', line=dict(color='orange')))
                fig_seir.add_trace(go.Scatter(x=pred_dates, y=predictions['Exposed'], name='Model Exposed', line=dict(dash='dot')))
                fig_seir.update_layout(title=f"SEIR Model Fit & Forecast - {selected_region}")
                st.plotly_chart(fig_seir, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {e}")
