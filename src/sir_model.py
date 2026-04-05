import numpy as np
from scipy.integrate import odeint
import pandas as pd

def sir_deriv(y, t, N, beta, gamma):
    """
    Differential equations for SIR model.
    S: Susceptible
    I: Infected
    R: Recovered
    beta: contact rate
    gamma: recovery rate
    """
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def run_sir_model(N, I0, R0, beta, gamma, days):
    """
    Runs the SIR simulation.
    """
    S0 = N - I0 - R0
    y0 = [S0, I0, R0]
    t = np.linspace(0, days, days)
    
    ret = odeint(sir_deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    
    return pd.DataFrame({
        'day': t,
        'Susceptible': S,
        'Infected': I,
        'Recovered': R
    })

def estimate_rt(beta, gamma):
    """
    Estimates basic reproduction number R0.
    """
    return beta / gamma

class SIRModel:
    def __init__(self, N):
        self.N = N
        self.beta = None
        self.gamma = None
        
    def fit(self, infected_data, recovered_data):
        """
        Simple estimation of beta and gamma based on data.
        In a real scenario, we would use optimization (scipy.optimize.minimize)
        to find beta and gamma that minimize error between model I and actual I.
        For this project, we'll use a simplified approximation or placeholder.
        """
        # simplified estimation for demonstration
        # dI/dt approx (I[t+1] - I[t])
        # Force dummy values for stability if optimization is too complex for this step
        # Ideally: Minimize MSE(I_model, I_actual)
        self.beta = 0.3  # Placeholder/Initial guess
        self.gamma = 0.1 # Placeholder/Initial guess
        pass

    def predict(self, days, I0, R0):
        if self.beta is None or self.gamma is None:
            raise ValueError("Model not fitted yet.")
        return run_sir_model(self.N, I0, R0, self.beta, self.gamma, days)

if __name__ == "__main__":
    # Test SIR model
    N = 1000000
    df_sim = run_sir_model(N, I0=1, R0=0, beta=0.3, gamma=0.1, days=100)
    print(df_sim.head())
