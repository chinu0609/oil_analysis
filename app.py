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

# Global variables
df = None
pred_arima = {"DL": [], "DS": []}
pred_exp = {"DL": [], "DS": []}
time_pred = []
next_time = 0
initial_window = 50
time_step = 0.5  # default time increment

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Forecast: ARIMA & Exponential Smoothing"

empty_dark_fig = go.Figure()
empty_dark_fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='#1e1e1e',
    plot_bgcolor='#1e1e1e',
    font=dict(color='white'),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False)
)

app.layout = dbc.Container([
    html.H2("📈 Continuous Forecasting Dashboard",
            className="text-center mb-4 text-white fw-bold"),

    # Controls
    dbc.Row([
        dbc.Col([
            dbc.Button("▶️ Start", id="toggle-btn", n_clicks=0, color="success", className="mb-3"),
            html.Label("⏳ Window Size", className="text-white"),
            dcc.Slider(
                id='window-slider', min=10, max=100, value=50, step=5,
                marks={i: str(i) for i in range(10, 110, 10)},
                tooltip={"placement": "bottom"},
                className="mb-4"
            ),
            html.Label("📏 Time Step", className="text-white"),
            dcc.Input(
                id='time-step-input', type='number', value=0.5, step=0.1, min=0.1,
                style={"width": "100px", "marginBottom": "1rem"},
                debounce=True
            )
        ], width=4)
    ]),

    # File Upload
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

    html.Div(id='status-text', className='text-white text-center mb-4 fw-semibold'),

    dcc.Interval(id='interval', interval=1000, n_intervals=0, disabled=True),

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
], fluid=True, style={'backgroundColor': '#1e1e1e', 'padding': '2rem'})


@app.callback(
    Output('status-text', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('time-step-input', 'value')
)
def update_output(contents, filename, ts_step):
    global df, time_pred, pred_arima, pred_exp, next_time, time_step
    if contents is None:
        return "No file uploaded yet."

    time_step = ts_step if ts_step else 0.5

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        time_vals = df.iloc[:, 0].tolist()
        dl_vals = df.iloc[:, 1].tolist()
        ds_vals = df.iloc[:, 2].tolist()

        # Use initial_window last values
        time_pred = [i * time_step for i in range(initial_window)]
        pred_arima = {"DL": dl_vals[-initial_window:], "DS": ds_vals[-initial_window:]}
        pred_exp = {"DL": dl_vals[-initial_window:], "DS": ds_vals[-initial_window:]}
        next_time = time_pred[-1] + time_step

        return f"✅ Loaded file: {filename} with shape {df.shape}"
    except Exception as e:
        return f"Error loading file: {str(e)}"


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


def forecast_next(method, series, window_size):
    if len(series) < window_size:
        window_size = len(series)

    try:
        if method == "arima":
            model = ARIMA(series, order=(2, 0, 1)).fit()
        else:
            model = ExponentialSmoothing(series, trend="add", seasonal=None,
                                          initialization_method="estimated").fit()
        return model.forecast()[0]
    except Exception as e:
        print(f"Error forecasting: {e}")
        return series[-1]


@app.callback(
    [Output('dl-graph', 'figure'),
     Output('ds-graph', 'figure'),
     Output('dlplusds-graph', 'figure')],
    Input('interval', 'n_intervals'),
    State('window-slider', 'value'),
    State('upload-data', 'contents'),
    State('time-step-input', 'value')
)
def update_forecast(n_intervals, window_size, uploaded_file, ts_step):
    global next_time, time_step
    if uploaded_file is None:
        return go.Figure(), go.Figure(), go.Figure()

    time_step = ts_step if ts_step else 0.5
    window_size = max(1, window_size)

    # Forecast next value
    dl_arima_next = forecast_next("arima", pred_arima["DL"][-window_size:], window_size)
    ds_arima_next = forecast_next("arima", pred_arima["DS"][-window_size:], window_size)
    dl_exp_next = forecast_next("exp", pred_exp["DL"][-window_size:], window_size)
    ds_exp_next = forecast_next("exp", pred_exp["DS"][-window_size:], window_size)

    pred_arima["DL"].append(dl_arima_next)
    pred_arima["DS"].append(ds_arima_next)
    pred_exp["DL"].append(dl_exp_next)
    pred_exp["DS"].append(ds_exp_next)
    time_pred.append(next_time)
    next_time += time_step

    dlplusds_arima = list(np.array(pred_arima["DL"]) + np.array(pred_arima["DS"]))
    dlplusds_exp = list(np.array(pred_exp["DL"]) + np.array(pred_exp["DS"]))

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

