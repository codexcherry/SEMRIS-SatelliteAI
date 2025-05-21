import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import json

class EnvironmentalDashboard:
    def __init__(self, data: dict, predictions: dict):
        """Initialize the dashboard with data and predictions."""
        self.data = data
        self.predictions = predictions
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Environmental Monitoring Dashboard", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
            
            # Region Summary Card
            html.Div([
                html.H2("Region Summary", style={'color': '#2c3e50'}),
                html.Div(id='region-summary-content')
            ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
            
            # Trend Visualization
            html.Div([
                html.H2("Historical and Predicted Trends", style={'color': '#2c3e50'}),
                dcc.Graph(id='trend-graph')
            ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
            
            # Risk Map
            html.Div([
                html.H2("Degradation Risk Map", style={'color': '#2c3e50'}),
                dcc.Graph(id='risk-map')
            ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
            
            # Restoration Priority
            html.Div([
                html.H2("Restoration Priority Areas", style={'color': '#2c3e50'}),
                dcc.Graph(id='priority-chart')
            ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}),
            
            # Hidden div for storing intermediate data
            html.Div(id='intermediate-data', style={'display': 'none'})
        ], style={'padding': '20px', 'backgroundColor': '#f5f6fa'})

    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        @self.app.callback(
            [Output('region-summary-content', 'children'),
             Output('trend-graph', 'figure'),
             Output('risk-map', 'figure'),
             Output('priority-chart', 'figure')],
            [Input('intermediate-data', 'children')]
        )
        def update_dashboard(_):
            # 1. Region Summary
            dataset = self.data['NDVI']
            pred = self.predictions['NDVI']
            current_avg = float(dataset['NDVI'].mean().values)
            predicted_avg = float(np.mean(pred))
            change_percent = ((predicted_avg - current_avg) / current_avg) * 100
            
            summary_content = html.Div([
                html.P(f"Current Average: {current_avg:.2f}"),
                html.P(f"Predicted Average: {predicted_avg:.2f}"),
                html.P(f"Change: {change_percent:.1f}%"),
                html.P(f"Trend: {'Improving' if change_percent > 0 else 'Degrading'}")
            ])
            
            # 2. Trend Graph
            dataset = dataset.resample(time='ME').mean()
            pred = pred[::60]  # Take every 60th prediction
            
            hist_data = dataset['NDVI'].mean(dim=['lat', 'lon']).values
            hist_dates = pd.to_datetime(dataset.time.values)
            pred_data = pred.mean(axis=(1, 2))
            last_date = hist_dates[-1]
            pred_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=len(pred_data),
                freq='2ME'
            )
            
            trend_fig = go.Figure()
            trend_fig.add_trace(go.Scatter(x=hist_dates, y=hist_data, name='Historical', line=dict(color='blue')))
            trend_fig.add_trace(go.Scatter(x=pred_dates, y=pred_data, name='Predicted', line=dict(color='red', dash='dash')))
            trend_fig.update_layout(
                title='NDVI Historical and Predicted Trends',
                xaxis_title='Date',
                yaxis_title='NDVI',
                template='plotly_white'
            )
            
            # 3. Risk Map
            pred = self._reduce_spatial_resolution(pred)
            risk_map = np.mean(pred, axis=0)
            risk_map = (risk_map - np.min(risk_map)) / (np.max(risk_map) - np.min(risk_map))
            
            risk_fig = px.imshow(
                risk_map,
                color_continuous_scale='RdYlGn_r',
                title='Degradation Risk Map',
                labels=dict(x='Longitude', y='Latitude', color='Risk Level')
            )
            
            # 4. Priority Chart
            priority_scores = np.mean(pred, axis=0)
            priority_scores = (priority_scores - np.min(priority_scores)) / (np.max(priority_scores) - np.min(priority_scores))
            
            lats = dataset.coords['lat'].values[::8]
            lons = dataset.coords['lon'].values[::8]
            lats_grid = np.repeat(lats, len(lons))
            lons_grid = np.tile(lons, len(lats))
            scores = priority_scores.reshape(-1)
            
            df = pd.DataFrame({
                'Latitude': lats_grid,
                'Longitude': lons_grid,
                'Priority_Score': scores
            })
            
            df = df.nlargest(20, 'Priority_Score')
            
            priority_fig = px.bar(
                df,
                x='Priority_Score',
                y='Latitude',
                title='Top 20 Areas by Restoration Priority',
                labels={'Priority_Score': 'Restoration Priority Score', 'Latitude': 'Location'},
                color='Priority_Score',
                color_continuous_scale='RdYlGn_r'
            )
            
            return summary_content, trend_fig, risk_fig, priority_fig

    def _reduce_spatial_resolution(self, data: np.ndarray, factor: int = 8) -> np.ndarray:
        """Reduce spatial resolution by averaging over blocks. Always returns a valid reduced array."""
        n_time, n_lat, n_lon = data.shape
        if n_lat < factor or n_lon < factor:
            return data.mean(axis=(1,2), keepdims=True)
        new_lat = n_lat // factor
        new_lon = n_lon // factor
        cropped_data = data[:, :new_lat*factor, :new_lon*factor]
        try:
            reshaped = cropped_data.reshape(n_time, new_lat, factor, new_lon, factor)
            return np.mean(reshaped, axis=(2, 4))
        except Exception:
            return data.mean(axis=(1,2), keepdims=True)

    def run(self, debug: bool = True, port: int = 8050):
        """Run the dashboard server."""
        self.app.run_server(debug=debug, port=port) 