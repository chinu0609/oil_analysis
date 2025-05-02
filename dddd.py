import pandas as pd
import time
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

def real_time_forecast(data, window_size, forecast_steps, delay=0.5):
    """
    Function to predict future values based on the past `window_size` values using 
    Exponential Smoothing and ARIMA models.
    Args:
        data: DataFrame with time, dl, and ds columns.
        window_size: Number of previous data points to use for prediction.
        forecast_steps: Number of steps to forecast into the future.
        delay: Delay in seconds between each prediction (simulating real-time).
    """
    # Ensure proper column names (replace with actual column names)
    time_col = data.columns[0]
    dl_col = data.columns[1]
    ds_col = data.columns[2]

    # Loop for real-time predictions
    while True:
        # Get the latest `window_size` data points
        window_data = data.iloc[-window_size:]

        # Separate columns
        time_vals = window_data[time_col].reset_index(drop=True)
        dl = window_data[dl_col].values
        ds = window_data[ds_col].values
        dl_plus_ds = dl + ds

        # Safe forecast function to avoid model failure
        def safe_forecast(model_class, data_series):
            try:
                model = model_class(data_series).fit()
                forecast = model.forecast(steps=forecast_steps)
                return forecast
            except Exception as e:
                print(f"Forecasting error: {e}")
                return [None] * forecast_steps

        # Forecasting using Exponential Smoothing and ARIMA
        dl_exp_forecast = safe_forecast(
            lambda x: ExponentialSmoothing(x, trend="add", seasonal=None, initialization_method="estimated"), dl)
        ds_exp_forecast = safe_forecast(
            lambda x: ExponentialSmoothing(x, trend="add", seasonal=None, initialization_method="estimated"), ds)
        dlds_exp_forecast = dl_exp_forecast + ds_exp_forecast

        dl_arima_forecast = safe_forecast(lambda x: ARIMA(x, order=(2, 0, 1)), dl)
        ds_arima_forecast = safe_forecast(lambda x: ARIMA(x, order=(2, 0, 1)), ds)
        dlds_arima_forecast = dl_arima_forecast + ds_arima_forecast

        # Print forecasts for current window
        print("Forecasts based on past values:")
        print("DL - Exponential Smoothing:", dl_exp_forecast)
        print("DS - Exponential Smoothing:", ds_exp_forecast)
        print("DL + DS - Exponential Smoothing:", dlds_exp_forecast)

        print("DL - ARIMA:", dl_arima_forecast)
        print("DS - ARIMA:", ds_arima_forecast)
        print("DL + DS - ARIMA:", dlds_arima_forecast)

        # Append forecasted values to the original data to update the window for the next iteration
        # Update the `time_col` and values by incrementing the time
        last_time = window_data[time_col].iloc[-1]
        next_time = last_time + 1  # Increment time for the next forecasted step

        # Append forecasted values (for next `forecast_steps`) to the window data
        for i in range(forecast_steps):
            data = data.append({
                time_col: next_time + i,
                dl_col: dl_exp_forecast[i] if dl_exp_forecast[i] is not None else dl_arima_forecast[i],
                ds_col: ds_exp_forecast[i] if ds_exp_forecast[i] is not None else ds_arima_forecast[i],
            }, ignore_index=True)

        # Simulate real-time delay
        time.sleep(delay)

# Example of how to use the function
if __name__ == "__main__":
    # Simulating some example data
    data = pd.DataFrame({
        'time': [i for i in range(1, 101)],
        'dl': [i + 10 for i in range(1, 101)],  # Sample DL data
        'ds': [i + 20 for i in range(1, 101)]   # Sample DS data
    })

    # Call the real-time forecast function
    real_time_forecast(data, window_size=20, forecast_steps=5, delay=1)
