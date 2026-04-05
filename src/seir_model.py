import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

def seir_deriv(y, t, N, beta, gamma, sigma):
    """
    Differential equations for SEIR model.
    S: Susceptible
    E: Exposed
    I: Infected
    R: Recovered
    beta: contact rate
    gamma: recovery rate
    sigma: incubation rate (1/incubation_period)
    """
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

def run_seir_model(N, I0, R0, E0, beta, gamma, sigma, days):
    """
    Runs the SEIR simulation.
    """
    S0 = N - I0 - R0 - E0
    y0 = [S0, E0, I0, R0]
    t = np.linspace(0, days, days)
    
    ret = odeint(seir_deriv, y0, t, args=(N, beta, gamma, sigma))
    S, E, I, R = ret.T
    
    return pd.DataFrame({
        'day': t,
        'Susceptible': S,
        'Exposed': E,
        'Infected': I,
        'Recovered': R
    })

class SEIRModel:
    def __init__(self, N):
        self.N = N
        self.beta = None
        self.gamma = None
        self.sigma = None
        self.metrics = {}
        
    def fit(self, infected_data, days_to_fit=None):
        """
        Fits beta, gamma, sigma to the infected data.
        infected_data: Time series of active infected cases.
        """
        if days_to_fit:
             # Fit only on the first N days if specified
             y_data = infected_data[:days_to_fit]
        else:
             y_data = infected_data

        days = len(y_data)
        t = np.linspace(0, days, days)
        
        # Initial guesses
        # gamma ~ 1/14 (recovery period 14 days)
        # sigma ~ 1/5 (incubation period 5 days)
        # beta > gamma for spread
        initial_guess = [0.2, 1.0/14, 1.0/5] 
        bounds = ((0, 2.0), (0, 1.0), (0, 1.0)) # beta, gamma, sigma bounds
        
        # Assume small start
        I0 = y_data[0] if len(y_data) > 0 and y_data[0] > 0 else 1
        E0 = I0 * 2 # Assumption
        R0 = 0
        S0 = self.N - I0 - E0 - R0
        
        def loss(params):
            b, g, s = params
            y0 = [S0, E0, I0, R0]
            ret = odeint(seir_deriv, y0, t, args=(self.N, b, g, s))
            I_pred = ret[:, 2] # Infected column
            return np.sqrt(mean_squared_error(y_data, I_pred))
            
        result = minimize(loss, initial_guess, bounds=bounds)
        self.beta, self.gamma, self.sigma = result.x
        
        self.metrics['RMSE'] = result.fun
        self.metrics['R0_est'] = self.beta / self.gamma
        
        return self.metrics

    def predict(self, days, I0=None):
        """
        Predicts future using fitted parameters.
        """
        if self.beta is None:
            raise ValueError("Model not fitted.")
        
        # If I0 not provided, use a small seed or infer from fit logic (simplified here)
        # In a real app, we should carry over state from the end of fitting
        if I0 is None:
            I0 = 1 # Fallback
            
        E0 = I0 * 2
        R0 = 0
        
        return run_seir_model(self.N, I0, R0, E0, self.beta, self.gamma, self.sigma, days)

if __name__ == "__main__":
    # Test
    N = 1_000_000
    model = SEIRModel(N)
    
    # Synthetic ground truth
    true_params = (0.3, 0.1, 0.2)
    df_true = run_seir_model(N, 10, 0, 20, *true_params, days=50)
    
    # Fit
    infected_data = df_true['Infected'].values + np.random.normal(0, 100, 50) # Add noise
    infected_data = np.clip(infected_data, 0, N)
    
    metrics = model.fit(infected_data)
    print(f"Fitted Metrics: {metrics}")
    print(f"Estimated Params: beta={model.beta:.3f}, gamma={model.gamma:.3f}, sigma={model.sigma:.3f}")
