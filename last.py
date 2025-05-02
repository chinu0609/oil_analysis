import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import base64
import io
import dash_bootstrap_components as dbc

# Global variables for prediction history and dataframe
df = None
pred_arima = {"DL": [], "DS": []}
pred_exp = {"DL": [], "DS": []}
time_pred = []
next_time = 0
initial_window = 50

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])  # or LUX, FLATLY, SLATE, etc.
app.title = "Forecast: ARIMA & Exponential Smoothing"

empty_dark_fig = go.Figure()
empty_dark_fig.update_layout(
    template='plotly_dark',           # sets dark background
    paper_bgcolor='#1e1e1e',          # background of outer chart
    plot_bgcolor='#1e1e1e',           # background of plotting area
    font=dict(color='white'),         # font color
    xaxis=dict(showgrid=False),       # optional: turn off grids
    yaxis=dict(showgrid=False)
)
app.layout = dbc.Container([
    html.H2("📈 Continuous Forecasting Dashboard", 
            className="text-center mb-4 text-white fw-bold"),

    # START + Slider
    dbc.Row([
        dbc.Col([
            dbc.Button("▶️ Start", id="toggle-btn", n_clicks=0, color="success", className="mb-3"),
            html.Label("⏳ Window Size", className="text-white"),
            dcc.Slider(
                id='window-slider', min=10, max=100, value=50, step=5,
                marks={i: str(i) for i in range(10, 110, 10)},
                tooltip={"placement": "bottom"},
                className="mb-4"
            )
        ], width=4)
    ]),

    # Upload Section
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            html.Button('📂 Drag & Drop or Browse File', 
                style={
                    'backgroundColor': '#4CAF50',
                    'color': 'white',
                    'padding': '12px 24px',
                    'border': 'none',
                    'borderRadius': '8px',
                    'cursor': 'pointer',
                    'fontSize': '16px',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                    'transition': '0.3s ease'
                })
        ]),
        style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'height': '120px',
            'borderWidth': '2px',
            'borderStyle': 'dashed',
            'borderRadius': '12px',
            'borderColor': '#4CAF50',
            'backgroundColor': '#2a2a2a',
            'textAlign': 'center',
            'marginBottom': '30px'
        },
        multiple=False
    ),

    # Status
    html.Div(id='status-text', className='text-white text-center mb-4 fw-semibold'),

    # Interval
    dcc.Interval(id='interval', interval=1000, n_intervals=0, disabled=True),

    # Forecast Cards
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("DL Forecast", className="bg-dark text-white"),
            dbc.CardBody(
                dcc.Graph(id='dl-graph', figure=empty_dark_fig, config={'displayModeBar': False}),
                className="bg-dark"
            )
        ], className="shadow-lg"), md=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("DS Forecast", className="bg-dark text-white"),
            dbc.CardBody(
                dcc.Graph(id='ds-graph', figure=empty_dark_fig, config={'displayModeBar': False}),
                className="bg-dark"
            )
        ], className="shadow-lg"), md=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("DL + DS Forecast", className="bg-dark text-white"),
            dbc.CardBody(
                dcc.Graph(id='dlplusds-graph', figure=empty_dark_fig, config={'displayModeBar': False}),
                className="bg-dark"
            )
        ], className="shadow-lg"), md=12)
    ])
], 
fluid=True, style={'backgroundColor': '#1e1e1e', 'padding': '2rem'})

# Function to update output text for file upload
@app.callback(
    Output('status-text', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    global df, time_pred, pred_arima, pred_exp, next_time
    if contents is None:
        return "No file uploaded yet."

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        time_vals = df.iloc[:, 0].tolist()
        dl_vals = df.iloc[:, 1].tolist()
        ds_vals = df.iloc[:, 2].tolist()

        time_pred = time_vals[:initial_window]
        pred_arima = {"DL": dl_vals[:initial_window], "DS": ds_vals[:initial_window]}
        pred_exp = {"DL": dl_vals[:initial_window], "DS": ds_vals[:initial_window]}
        next_time = time_pred[-1] + 1 if isinstance(time_pred[-1], (int, float)) else initial_window

        return f"✅ Loaded file: {filename} with shape {df.shape}"
    except Exception as e:
        return f"Error loading file: {str(e)}"

# Function to toggle the interval for continuous prediction
@app.callback(
    Output('interval', 'disabled'),
    Output('toggle-btn', 'children'),
    Input('toggle-btn', 'n_clicks'),
    State('interval', 'disabled')
)
def toggle_interval(n_clicks, disabled):
    if n_clicks == 0:
        return dash.no_update, dash.no_update
    return not disabled, "Stop" if disabled else "Start"

# Function to forecast next value using ARIMA or Exponential Smoothing
# Function to forecast next value using ARIMA or Exponential Smoothing
def forecast_next(method, series, window_size):
    if len(series) < window_size:
        # If series is smaller than the window size, use whatever data is available
        print(f"Warning: Series length is smaller than window_size. Using available data of length {len(series)}.")
        window_size = len(series)  # Adjust window size to be the series length

    try:
        if method == "arima":
            model = ARIMA(series, order=(2, 0, 1)).fit()
        else:
            model = ExponentialSmoothing(series, trend="add", seasonal=None,
                                          initialization_method="estimated").fit()
        return model.forecast()[0]
    except Exception as e:
        print(f"Error forecasting: {e}")
        return series[-1]  # Return last value if forecasting fails

from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import plotly.graph_objects as go

@app.callback(
    [Output('dl-graph', 'figure'),
     Output('ds-graph', 'figure'),
     Output('dlplusds-graph', 'figure')],
    [Input('interval', 'n_intervals')],
    [State('window-slider', 'value'),
     State('upload-data', 'contents')]  # Checking the file upload
)
def update_forecast(n_intervals, window_size, uploaded_file):
    global next_time
    if uploaded_file is None:
        # If no file is uploaded, return empty figures or placeholder message
        return go.Figure(), go.Figure(), go.Figure()

    # Ensure the window size is at least 1
    window_size = max(1, window_size)

    # Forecast new values using ARIMA and Exponential Smoothing
    dl_arima_next = forecast_next("arima", pred_arima["DL"][-window_size:], window_size)
    ds_arima_next = forecast_next("arima", pred_arima["DS"][-window_size:], window_size)
    dl_exp_next = forecast_next("exp", pred_exp["DL"][-window_size:], window_size)
    ds_exp_next = forecast_next("exp", pred_exp["DS"][-window_size:], window_size)

    # Append predictions to the history
    pred_arima["DL"].append(dl_arima_next)
    pred_arima["DS"].append(ds_arima_next)
    pred_exp["DL"].append(dl_exp_next)
    pred_exp["DS"].append(ds_exp_next)
    time_pred.append(next_time)
    next_time += 1

    # DL + DS combined forecast
    dlplusds_arima = list(np.array(pred_arima["DL"]) + np.array(pred_arima["DS"]))
    dlplusds_exp = list(np.array(pred_exp["DL"]) + np.array(pred_exp["DS"]))

    # Create the figures for DL, DS, and DL + DS
    fig_dl = go.Figure()
    fig_dl.add_trace(go.Scatter(x=time_pred, y=pred_arima["DL"], mode='lines',
                                name="ARIMA", line=dict(color='blue', dash="dot")))
    fig_dl.add_trace(go.Scatter(x=time_pred, y=pred_exp["DL"], mode='lines',
                                name="Exp Smoothing", line=dict(color='blue')))
    fig_dl.update_layout(template="plotly_dark", title="DL Prediction", xaxis_title="Time", yaxis_title="DL Value")

    fig_ds = go.Figure()
    fig_ds.add_trace(go.Scatter(x=time_pred, y=pred_arima["DS"], mode='lines',
                                name="ARIMA", line=dict(color='green', dash="dot")))
    fig_ds.add_trace(go.Scatter(x=time_pred, y=pred_exp["DS"], mode='lines',
                                name="Exp Smoothing", line=dict(color='green')))
    fig_ds.update_layout(template="plotly_dark", title="DS Prediction", xaxis_title="Time", yaxis_title="DS Value")

    fig_dlplusds = go.Figure()
    fig_dlplusds.add_trace(go.Scatter(x=time_pred, y=dlplusds_arima, mode='lines',
                                      name="ARIMA", line=dict(color='purple', dash="dot")))
    fig_dlplusds.add_trace(go.Scatter(x=time_pred, y=dlplusds_exp, mode='lines',
                                      name="Exp Smoothing", line=dict(color='purple')))
    fig_dlplusds.update_layout(template="plotly_dark", title="DL + DS Prediction", xaxis_title="Time", yaxis_title="DL + DS Value")

    return fig_dl, fig_ds, fig_dlplusds

if __name__ == "__main__":
    app.run(debug=True)
