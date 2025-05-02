import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

# Setup Streamlit page
st.set_page_config(page_title="Continuous Real-Time Forecasting", layout="wide")
st.title("Continuous Real-Time Particle Forecasting: ARIMA & Exponential Smoothing")

# File upload
uploaded_file = st.file_uploader("Upload your particle data CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Sample Data", data.head())

    # Required columns
    time_col = data.columns[0]
    dl_col = data.columns[1]
    ds_col = data.columns[2]

    # Ensure time values are floating point
    try:
        data[time_col] = data[time_col].astype(float)
    except:
        st.warning(f"Time column '{time_col}' could not be converted to float. Make sure it's numerical.")
        st.stop()

    # Set initial parameters
    window_size = st.slider("Sliding window size", min_value=10, max_value=200, value=50, step=5)
    forecast_steps = st.slider("Forecast steps", min_value=1, max_value=50, value=10)
    delay = st.slider("Delay between updates (s)", 0.1, 2.0, 0.3, 0.1)

    # Initial setup: Start with a sliding window of data points
    window_data = data.iloc[:window_size]
    time_vals = window_data[time_col].reset_index(drop=True)
    dl = window_data[dl_col].values
    ds = window_data[ds_col].values
    dl_plus_ds = dl + ds

    # Forecasting loop (Real-time continuous predictions)
    plot_placeholder = st.empty()
    while True:
        # Update the window with new data if available or use predictions for further steps
        if len(data) > len(window_data):
            window_data = data.iloc[len(window_data) - window_size:len(window_data)]
        else:
            last_time = window_data[time_col].iloc[-1]
            last_dl = window_data[dl_col].iloc[-1]
            last_ds = window_data[ds_col].iloc[-1]
            window_data = window_data.append({
                time_col: last_time + 1,  # Simulate the next time step
                dl_col: last_dl,  # Continue the last value for now
                ds_col: last_ds   # Continue the last value for now
            }, ignore_index=True)

        time_vals = window_data[time_col].reset_index(drop=True)
        dl = window_data[dl_col].values
        ds = window_data[ds_col].values
        dl_plus_ds = dl + ds

        # Forecast the next 'forecast_steps' time points based on the last time point
        last_time = time_vals.iloc[-1]
        forecast_time = [last_time + (j + 1) for j in range(forecast_steps)]

        # Safe forecast function to avoid model failure
        def safe_forecast(model_class, data_series):
            try:
                model = model_class(data_series).fit()
                forecast = model.forecast(steps=forecast_steps)
                return forecast
            except Exception as e:
                st.warning(f"Forecasting error: {e}")
                return [None] * forecast_steps

        # Forecasts
        dl_exp_forecast = safe_forecast(
            lambda x: ExponentialSmoothing(x, trend="add", seasonal=None, initialization_method="estimated"), dl)
        ds_exp_forecast = safe_forecast(
            lambda x: ExponentialSmoothing(x, trend="add", seasonal=None, initialization_method="estimated"), ds)
        dlds_exp_forecast = dl_exp_forecast + ds_exp_forecast

        dl_arima_forecast = safe_forecast(lambda x: ARIMA(x, order=(2, 0, 1)), dl)
        ds_arima_forecast = safe_forecast(lambda x: ARIMA(x, order=(2, 0, 1)), ds)
        dlds_arima_forecast = dl_arima_forecast + ds_arima_forecast

        # Plotting utility
        def plot_forecast(title, orig, forecast, ylabel, color, time_vals):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time_vals, orig, label="Original", color=color, alpha=0.8)
            ax.plot(forecast_time, forecast, label="Forecast", color=color, linestyle="--", alpha=0.5)
            ax.set_title(title)
            ax.set_xlabel("Time")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)
            return fig

        # Display the forecast
        with plot_placeholder.container():
            st.subheader(f"Continuous Real-Time Predictions")

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_forecast("DL - Exponential Smoothing", dl, dl_exp_forecast, "DL", "blue", time_vals))
            with col2:
                st.pyplot(plot_forecast("DL - ARIMA", dl, dl_arima_forecast, "DL", "blue", time_vals))

            col3, col4 = st.columns(2)
            with col3:
                st.pyplot(plot_forecast("DS - Exponential Smoothing", ds, ds_exp_forecast, "DS", "green", time_vals))
            with col4:
                st.pyplot(plot_forecast("DS - ARIMA", ds, ds_arima_forecast, "DS", "green", time_vals))

            col5, col6 = st.columns(2)
            with col5:
                st.pyplot(plot_forecast("DL+DS - Exponential Smoothing", dl_plus_ds, dlds_exp_forecast, "DL+DS", "purple", time_vals))
            with col6:
                st.pyplot(plot_forecast("DL+DS - ARIMA", dl_plus_ds, dlds_arima_forecast, "DL+DS", "purple", time_vals))

        # Wait for the next iteration (simulating real-time delay)
        time.sleep(delay)
