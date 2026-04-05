from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculates MAE and RMSE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        'MAE': mae,
        'RMSE': rmse
    }

if __name__ == "__main__":
    y_true = [1, 2, 3, 4]
    y_pred = [1.1, 1.9, 3.2, 3.8]
    print(calculate_metrics(y_true, y_pred))
