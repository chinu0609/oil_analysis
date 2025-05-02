import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

# Streamlit page setup
st.set_page_config(page_title="Continuous Forecast", layout="wide")
st.title("Continuous Real-Time Forecasting using ARIMA & Exponential Smoothing")

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'data' not in st.session_state:
    st.session_state.data = None

# Upload file
uploaded_file = st.file_uploader("Upload your particle data CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.session_state.data = data

if st.session_state.data is not None:
    data = st.session_state.data
    st.write("### Sample Data", data.head())

    # Checking if the dataset has necessary columns
    if data.shape[1] < 3:
        st.error("Data must have at least 3 columns for time, dl, and ds.")
        st.stop()

    # Extracting column names
    time_col, dl_col, ds_col = data.columns[:3]

    # Convert time column to float if possible
    try:
        data[time_col] = data[time_col].astype(float)
    except:
        st.error("Time column must be convertible to float.")
        st.stop()

    # Parameters
    window_size = st.slider("Window size (past points to use)", 10, 200, 50, 5)
    forecast_steps = st.slider("Forecast steps", 1, 50, 10)
    delay = st.slider("Delay between updates (s)", 0.1, 2.0, 0.3, 0.1)

    # Simulate incoming data
    current_data = data.copy()
    if st.session_state.step >= len(current_data):
        last_row = current_data.iloc[-1]
        new_row = {
            time_col: last_row[time_col] + 1,
            dl_col: last_row[dl_col],
            ds_col: last_row[ds_col],
        }
        current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state.data = current_data

    # Sliding window selection
    if len(current_data) < window_size:
        st.warning("Not enough data points to apply the window.")
        st.stop()

    window_data = current_data.iloc[-window_size:]
    time_vals = window_data[time_col].values
    dl_vals = window_data[dl_col].values
    ds_vals = window_data[ds_col].values
    dlds_vals = dl_vals + ds_vals
    last_time = time_vals[-1]
    forecast_time = [last_time + i + 1 for i in range(forecast_steps)]

    # Forecasting wrapper
    def safe_forecast(model_fn, series):
        try:
            model = model_fn(series).fit()
            return model.forecast(steps=forecast_steps)
        except Exception as e:
            st.warning(f"Forecasting error: {e}")
            return [None] * forecast_steps

    # Forecasts
    dl_exp_forecast = safe_forecast(
        lambda x: ExponentialSmoothing(x, trend="add", seasonal=None, initialization_method="estimated"), dl_vals)
    ds_exp_forecast = safe_forecast(
        lambda x: ExponentialSmoothing(x, trend="add", seasonal=None, initialization_method="estimated"), ds_vals)
    dlds_exp_forecast = dl_exp_forecast + ds_exp_forecast

    dl_arima_forecast = safe_forecast(lambda x: ARIMA(x, order=(2, 0, 1)), dl_vals)
    ds_arima_forecast = safe_forecast(lambda x: ARIMA(x, order=(2, 0, 1)), ds_vals)
    dlds_arima_forecast = dl_arima_forecast + ds_arima_forecast

    # Plotting function
    def plot_forecast(title, original, forecast, ylabel, color):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(time_vals, original, label="Original", color=color, alpha=0.7)
        ax.plot(forecast_time, forecast, label="Forecast", color=color, linestyle="--", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)  # Important to prevent stale plots

    st.subheader(f"Real-Time Forecasts (Step {st.session_state.step})")
    col1, col2 = st.columns(2)
    with col1:
        plot_forecast("DL - Exponential Smoothing", dl_vals, dl_exp_forecast, "DL", "blue")
    with col2:
        plot_forecast("DL - ARIMA", dl_vals, dl_arima_forecast, "DL", "blue")

    col3, col4 = st.columns(2)
    with col3:
        plot_forecast("DS - Exponential Smoothing", ds_vals, ds_exp_forecast, "DS", "green")
    with col4:
        plot_forecast("DS - ARIMA", ds_vals, ds_arima_forecast, "DS", "green")

    col5, col6 = st.columns(2)
    with col5:
        plot_forecast("DL+DS - Exponential Smoothing", dlds_vals, dlds_exp_forecast, "DL+DS", "purple")
    with col6:
        plot_forecast("DL+DS - ARIMA", dlds_vals, dlds_arima_forecast, "DL+DS", "purple")

    # Delay and rerun
    st.session_state.step += 1
    time.sleep(delay)
    st.rerun()
