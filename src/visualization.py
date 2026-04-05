import matplotlib.pyplot as plt
import plotly.express as px
import os

def save_plot(fig, filename, output_dir='outputs'):
    """
    Saves a plot to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    print(f"Saved plot to {filepath}")

def plot_epidemic_curve(df, region):
    """
    Plots the epidemic curve (Confirmed, Deaths, Recovered).
    """
    plt.figure(figsize=(10, 6))
    region_data = df[df['region'] == region]
    plt.plot(region_data['date'], region_data['confirmed_cases'], label='Confirmed')
    plt.plot(region_data['date'], region_data['deaths'], label='Deaths')
    plt.plot(region_data['date'], region_data['recovered'], label='Recovered')
    plt.title(f'Epidemic Curve for {region}')
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.legend()
    plt.grid(True)
    save_plot(plt, f'{region}_epidemic_curve.png')
    plt.close()

def plot_growth_rate(df, region):
    """
    Plots daily growth rate.
    """
    plt.figure(figsize=(10, 6))
    region_data = df[df['region'] == region]
    plt.plot(region_data['date'], region_data['daily_growth_rate'], label='Daily Growth Rate', color='purple')
    plt.title(f'Daily Growth Rate for {region}')
    plt.xlabel('Date')
    plt.ylabel('Growth Rate')
    plt.legend()
    plt.grid(True)
    save_plot(plt, f'{region}_growth_rate.png')
    plt.close()

def plot_forecast(train_date, train_actual, test_date, test_actual, forecast, model_name, region):
    """
    Plots actual vs forecast.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train_date, train_actual, label='Train (Actual)')
    plt.plot(test_date, test_actual, label='Test (Actual)', color='green')
    plt.plot(test_date, forecast, label=f'Forecast ({model_name})', color='red', linestyle='--')
    plt.title(f'{model_name} Forecast for {region}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    save_plot(plt, f'{region}_{model_name}_forecast.png')
    plt.close()
