import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time
import threading

# Initialize the app
app = dash.Dash(__name__)

# Global variables for managing data and prediction state
df = pd.DataFrame()
predictions = []
is_predicting = False
prediction_thread = None

# App layout
app.layout = html.Div([
    html.H1("Real-time Prediction and Forecasting"),
    
    # File upload
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload CSV'),
        multiple=False
    ),
    
    html.Br(),

    # Sliders for adjusting parameters
    dcc.Slider(
        id='window-size',
        min=1,
        max=50,
        step=1,
        value=10,
        marks={i: str(i) for i in range(1, 51, 5)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    dcc.Slider(
        id='forecast-steps',
        min=1,
        max=20,
        step=1,
        value=5,
        marks={i: str(i) for i in range(1, 21, 5)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    dcc.Slider(
        id='delay',
        min=1,
        max=5,
        step=1,
        value=1,
        marks={i: str(i) for i in range(1, 6)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    
    # Buttons for control
    html.Button('Start Prediction', id='start-btn', n_clicks=0),
    html.Button('Stop Prediction', id='stop-btn', n_clicks=0),

    html.Div(id='output-data-upload'),
    
    # Graph to display predictions
    dcc.Graph(id='live-update-graph'),
])

# Callback to handle file upload
@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents')
)
def update_output(content):
    global df
    if content:
        # Decode and load the CSV
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Assuming the CSV has no header
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return html.Div([html.H5(f"Data loaded with shape: {df.shape}")])
        except Exception as e:
            return html.Div([f'Error processing file: {str(e)}'])

# Prediction function to simulate real-time forecasting
def prediction_loop(window_size, forecast_steps, delay):
    global is_predicting, predictions
    # Example: dummy forecasting using a rolling average
    while is_predicting:
        # Simulating prediction logic (replace with your model's prediction)
        if df.shape[0] > window_size:
            prediction = df.iloc[-window_size:].mean()  # Dummy prediction using a simple moving average
            predictions.append(prediction)
        
        # Create prediction time series (dummy)
        time_vals = list(range(len(df)))  # Simulating time-based index
        forecast_time = [time_vals[-1] + (i + 1) for i in range(forecast_steps)]
        
        # Plot data
        fig = go.Figure(data=[
            go.Scatter(x=time_vals, y=df.iloc[:, 0], mode='lines', name='Actual Data'),
            go.Scatter(x=forecast_time, y=predictions[-forecast_steps:], mode='lines', name='Predictions')
        ])
        
        # Update graph
        app.layout['live-update-graph'].figure = fig
        
        time.sleep(delay)  # Simulate real-time delay

# Callback to handle starting and stopping predictions
@app.callback(
    [Output('start-btn', 'disabled'),
     Output('stop-btn', 'disabled'),
     Output('live-update-graph', 'figure')],
    [Input('start-btn', 'n_clicks'),
     Input('stop-btn', 'n_clicks'),
     Input('window-size', 'value'),
     Input('forecast-steps', 'value'),
     Input('delay', 'value')]
)
def control_prediction(start_clicks, stop_clicks, window_size, forecast_steps, delay):
    global is_predicting, prediction_thread

    if start_clicks > 0 and not is_predicting:
        # Start prediction loop
        is_predicting = True
        prediction_thread = threading.Thread(target=prediction_loop, args=(window_size, forecast_steps, delay))
        prediction_thread.start()

    if stop_clicks > 0 and is_predicting:
        # Stop prediction loop
        is_predicting = False
        prediction_thread.join()

    # Disable start/stop buttons
    return (start_clicks == 0, stop_clicks == 0, {})

if __name__ == '__main__':
    app.run(debug=True)
