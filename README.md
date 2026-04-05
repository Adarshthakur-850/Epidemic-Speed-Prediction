<<<<<<< HEAD
# Epidemic Spread Speed Prediction

## Project Overview
This project models and predicts the speed of epidemic spread using time-series data and compartmental models (SIR). It includes modules for data loading, preprocessing, feature engineering, epidemiological modeling, forecasting (ARIMA/LSTM), and evaluation.

## Strict Requirements Implementation

- **Project Architecture**: Modular structure with `src/` directory.
- **Data Engineering**: Handles missing values, date parsing, and normalization.
- **Feature Engineering**: Calculates growth rates, rolling averages, and lags.
- **Modeling**: SIR model implementation and Rt estimation.
- **Forecasting**: ARIMA and LSTM models.
- **Evaluation**: MAE, RMSE metrics.
- **Visualization**: Epidemic curves, heatmaps (if region data available), and forecast plots.

## Project Structure
```
Epidemic Speed Prediction/
├── models/               # Saved models
├── outputs/              # Saved plots and results
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── sir_model.py
│   ├── forecasting.py
│   ├── evaluation.py
│   └── visualization.py
├── main.py               # Main pipeline execution
├── requirements.txt
└── README.md
```

## How to Run Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Pipeline**:
   ```bash
   python main.py
   ```
   This will generate a synthetic dataset (if no data is provided), train models, and save outputs to `outputs/` and `models/`.

## Future Extensions
- **FastAPI**: The structure is ready for API integration using the `src` modules.
- **Docker**: Can be containerized using a `Dockerfile`.
=======
# Epidemic-Speed-Prediction
ml project
>>>>>>> 6bb05fdea84c538e4f062cfb114c3af994e64252
