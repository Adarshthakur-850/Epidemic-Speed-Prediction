from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def train_arima(data, order=(5,1,0)):
    """
    Trains an ARIMA model.
    data: 1D array-like time series
    """
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

def create_sequences(data, seq_length):
    """
    Creates sequences for LSTM.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def train_lstm(data, seq_length=10, epochs=10, batch_size=1):
    """
    Trains an LSTM model for growth rate prediction.
    """
    # Scale data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    # Create sequences
    X, y = create_sequences(data_scaled, seq_length)
    
    # Reshape X for LSTM [samples, time steps, features]
    # X is already [samples, seq_length, 1] due to scaler fit_transform
    
    # Build model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    
    return model, scaler

def predict_arima(model_fit, steps=10):
    return model_fit.forecast(steps=steps)

def predict_lstm(model, scaler, last_sequence, steps=10):
    """
    Iterative forecasting with LSTM.
    last_sequence: shape (seq_length, 1) scaled
    """
    forecast = []
    current_seq = last_sequence.copy()
    
    for _ in range(steps):
        # Predict next value
        # Reshape to [1, seq_length, 1]
        pred = model.predict(current_seq.reshape(1, len(current_seq), 1), verbose=0)
        forecast.append(pred[0, 0])
        
        # Update sequence: shift left and append prediction
        current_seq = np.roll(current_seq, -1)
        current_seq[-1] = pred[0, 0]
        
    return scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
