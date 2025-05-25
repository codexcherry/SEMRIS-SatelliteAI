import os
from flask import Flask, render_template, send_from_directory
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import xarray as xr
from typing import Dict
import json
from datetime import datetime, timedelta

class WebInterface:
    def __init__(self, data: Dict[str, xr.Dataset], predictions: Dict[str, np.ndarray]):
        """
        Initialize the web interface with data and predictions for all parameters.
        
        Args:
            data: Dictionary of xarray Datasets (key: parameter name)
            predictions: Dictionary of numpy arrays (key: parameter name)
        """
        self.data = data
        self.predictions = predictions
        self.app = Flask(__name__, 
                       template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'templates'),
                       static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'static'))
        
        # Create necessary directories if they don't exist
        os.makedirs(self.app.template_folder, exist_ok=True)
        os.makedirs(self.app.static_folder, exist_ok=True)
        
        # Create index.html template if it doesn't exist
        self._create_template()
        
        # Add favicon route
        @self.app.route('/favicon.ico')
        def favicon():
            return send_from_directory(self.app.static_folder,
                                     'favicon.ico', mimetype='image/vnd.microsoft.icon')
        
        self.setup_routes()

    def _create_template(self):
        """Create the HTML template file if it doesn't exist."""
        template_path = os.path.join(self.app.template_folder, 'index.html')
        
        if not os.path.exists(template_path):
            with open(template_path, 'w') as f:
                f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SEMRIS - Satellite Environmental Monitoring</title>
    <link rel="icon" href="/favicon.ico">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background-color: #007bff;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .parameter-selector {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }
        .parameter-selector button {
            padding: 10px 15px;
            margin: 0 5px;
            border: none;
            background-color: #e9ecef;
            cursor: pointer;
            border-radius: 5px;
        }
        .parameter-selector button.active {
            background-color: #007bff;
            color: white;
        }
        .card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .summary-card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
            text-align: center;
        }
        .summary-card h3 {
            margin-top: 0;
            color: #007bff;
        }
        .summary-card .value {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .summary-card .change {
            color: #28a745;
        }
        .summary-card .change.negative {
            color: #dc3545;
        }
        .visualization-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }
        .visualization-card {
            height: 400px;
        }
        h2 {
            color: #333;
            margin-top: 0;
        }
        @media (max-width: 768px) {
            .visualization-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Satellite Environmental Monitoring and Restoration Intelligence System</h1>
            <p>Bengaluru Region Analysis</p>
        </div>
        
        <div class="parameter-selector">
            {% for param in parameters %}
            <button class="param-btn {% if param == 'NDVI' %}active{% endif %}" data-param="{{ param }}">{{ param }}</button>
            {% endfor %}
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Current Average</h3>
                <div class="value" id="current-avg">{{ summary.current_avg }}</div>
            </div>
            <div class="summary-card">
                <h3>Predicted Average</h3>
                <div class="value" id="predicted-avg">{{ summary.predicted_avg }}</div>
            </div>
            <div class="summary-card">
                <h3>Change</h3>
                <div class="value change {% if '-' in summary.change_percent %}negative{% endif %}" id="change-percent">
                    {{ summary.change_percent }}
                </div>
            </div>
            <div class="summary-card">
                <h3>Trend</h3>
                <div class="value" id="trend">{{ summary.trend }}</div>
            </div>
        </div>
        
        <div class="visualization-grid">
            <div class="card visualization-card">
                <h2>Historical and Predicted Trends</h2>
                <div id="trend-graph"></div>
            </div>
            <div class="card visualization-card">
                <h2>Degradation Risk Map</h2>
                <div id="risk-map"></div>
            </div>
            <div class="card visualization-card">
                <h2>Restoration Priority Areas</h2>
                <div id="priority-chart"></div>
            </div>
            <div class="card visualization-card">
                <h2>Multi-parameter Analysis</h2>
                <div id="multi-param-chart"></div>
            </div>
        </div>
    </div>

    <script>
        // Initialize plots
        var trendGraph = {{ trend_graph | safe }};
        var riskMap = {{ risk_map | safe }};
        var priorityChart = {{ priority_chart | safe }};
        var multiParamChart = {{ multi_param_chart | safe }};
        
        Plotly.newPlot('trend-graph', trendGraph.data, trendGraph.layout);
        Plotly.newPlot('risk-map', riskMap.data, riskMap.layout);
        Plotly.newPlot('priority-chart', priorityChart.data, priorityChart.layout);
        Plotly.newPlot('multi-param-chart', multiParamChart.data, multiParamChart.layout);
        
        // Parameter selector functionality
        document.querySelectorAll('.param-btn').forEach(button => {
            button.addEventListener('click', function() {
                const param = this.getAttribute('data-param');
                
                // Update active button
                document.querySelectorAll('.param-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                this.classList.add('active');
                
                // Fetch and update visualizations
                fetch(`/api/parameter/${param}`)
                    .then(response => response.json())
                    .then(data => {
                        // Update summary
                        document.getElementById('current-avg').textContent = data.summary.current_avg;
                        document.getElementById('predicted-avg').textContent = data.summary.predicted_avg;
                        document.getElementById('change-percent').textContent = data.summary.change_percent;
                        document.getElementById('trend').textContent = data.summary.trend;
                        
                        if (data.summary.change_percent.includes('-')) {
                            document.getElementById('change-percent').classList.add('negative');
                        } else {
                            document.getElementById('change-percent').classList.remove('negative');
                        }
                        
                        // Update plots
                        Plotly.react('trend-graph', data.trend_graph.data, data.trend_graph.layout);
                        Plotly.react('risk-map', data.risk_map.data, data.risk_map.layout);
                        Plotly.react('priority-chart', data.priority_chart.data, data.priority_chart.layout);
                        Plotly.react('multi-param-chart', data.multi_param_chart.data, data.multi_param_chart.layout);
                    });
            });
        });
    </script>
</body>
</html>
                """)

    def setup_routes(self):
        """Setup the web routes."""
        @self.app.route('/')
        def index():
            # Get list of all parameters
            parameters = list(self.data.keys())
            default_param = 'NDVI' if 'NDVI' in parameters else parameters[0]
            
            # Generate visualizations for default parameter
            trend_graph = self._create_trend_graph(default_param)
            risk_map = self._create_risk_map(default_param)
            priority_chart = self._create_priority_chart(default_param)
            multi_param_chart = self._create_multi_parameter_chart(default_param)
            summary = self._create_summary(default_param)

            return render_template('index.html',
                                parameters=parameters,
                                trend_graph=trend_graph,
                                risk_map=risk_map,
                                priority_chart=priority_chart,
                                multi_param_chart=multi_param_chart,
                                summary=summary)
        
        @self.app.route('/api/parameter/<parameter>')
        def parameter_data(parameter):
            """API endpoint to get data for a specific parameter."""
            if parameter not in self.data:
                return json.dumps({"error": f"Parameter {parameter} not found"}), 404
            
            trend_graph = self._create_trend_graph(parameter)
            risk_map = self._create_risk_map(parameter)
            priority_chart = self._create_priority_chart(parameter)
            multi_param_chart = self._create_multi_parameter_chart(parameter)
            summary = self._create_summary(parameter)
            
            return json.dumps({
                "trend_graph": json.loads(trend_graph),
                "risk_map": json.loads(risk_map),
                "priority_chart": json.loads(priority_chart),
                "multi_param_chart": json.loads(multi_param_chart),
                "summary": summary
            })

    def _create_trend_graph(self, parameter: str) -> str:
        """Create trend visualization for a parameter."""
        dataset = self.data[parameter]
        pred = self.predictions[parameter]
        
        try:
            # Process data - handle potential errors
            dataset = dataset.resample(time='ME').mean()
            
            # Take a reasonable number of prediction points
            step = max(1, len(pred) // 10)  # Ensure we don't take too many or too few points
            pred_subset = pred[::step]
            
            # Prepare data for plotting
            hist_data = dataset[parameter].mean(dim=['lat', 'lon']).values
            hist_dates = pd.to_datetime(dataset.time.values)
            
            # Ensure we have data
            if len(hist_data) == 0 or len(pred_subset) == 0:
                raise ValueError("Not enough data points for visualization")
                
            pred_data = pred_subset.mean(axis=(1, 2))
            last_date = hist_dates[-1]
            
            # Create prediction dates with appropriate frequency
            pred_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=len(pred_data),
                freq='1ME'
            )
            
            # Create figure with clear styling
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_dates, 
                y=hist_data, 
                name='Historical', 
                line=dict(color='#1f77b4', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=pred_dates, 
                y=pred_data, 
                name='Predicted', 
                line=dict(color='#d62728', width=3, dash='dash')
            ))
            
            # Improve layout
            fig.update_layout(
                title=f'{parameter} Historical and Predicted Trends',
                xaxis_title='Date',
                yaxis_title=parameter,
                template='plotly_white',
                height=350,
                margin=dict(l=50, r=50, t=50, b=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            # Create fallback visualization if there's an error
            print(f"Error creating trend graph: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Could not generate trend visualization for {parameter}.<br>Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(
                title=f'{parameter} Historical and Predicted Trends',
                height=350,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def _create_risk_map(self, parameter: str) -> str:
        """Create the risk map visualization."""
        try:
            pred = self.predictions[parameter]
            
            # Ensure we have enough data
            if pred.shape[0] == 0:
                raise ValueError("No prediction data available")
                
            # Get spatial resolution that works with the data
            pred = self._reduce_spatial_resolution(pred)
            
            # Calculate risk as the mean of predictions
            risk_map = np.mean(pred, axis=0)
            
            # Normalize risk map for visualization
            if np.max(risk_map) != np.min(risk_map):
                risk_map = (risk_map - np.min(risk_map)) / (np.max(risk_map) - np.min(risk_map))
            
            # Create a better color scale based on parameter
            if parameter == 'NDVI':
                color_scale = 'RdYlGn'  # Red (low NDVI) to Green (high NDVI)
            elif parameter == 'LST':
                color_scale = 'RdBu_r'  # Red (high temp) to Blue (low temp)
            elif parameter == 'Precipitation':
                color_scale = 'Blues'   # Blues for precipitation
            elif parameter == 'Biomass':
                color_scale = 'Greens'  # Greens for biomass
            else:
                color_scale = 'Viridis' # Default colorscale
            
            # Create heatmap with improved styling
            fig = px.imshow(
                risk_map,
                color_continuous_scale=color_scale,
                title=f'{parameter} Spatial Distribution',
                labels=dict(x='Longitude', y='Latitude', color=parameter),
                height=350,
                aspect='equal'
            )
            
            # Add colorbar title
            fig.update_coloraxes(colorbar_title=parameter)
            
            # Better layout
            fig.update_layout(
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            # Fallback visualization
            print(f"Error creating risk map: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Could not generate risk map for {parameter}.<br>Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(
                title=f'{parameter} Spatial Distribution',
                height=350,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def _create_priority_chart(self, parameter: str) -> str:
        """Create the priority chart visualization."""
        try:
            dataset = self.data[parameter]
            pred = self.predictions[parameter]
            
            # Ensure we have enough data
            if pred.shape[0] == 0:
                raise ValueError("No prediction data available")
                
            # Get spatial resolution that works with the data
            pred = self._reduce_spatial_resolution(pred)
            
            # Calculate priority scores
            priority_scores = np.mean(pred, axis=0)
            
            # Normalize priority scores
            if np.max(priority_scores) != np.min(priority_scores):
                priority_scores = (priority_scores - np.min(priority_scores)) / (np.max(priority_scores) - np.min(priority_scores))
            
            # Get coordinates and ensure they match the priority_scores shape
            n_lat, n_lon = priority_scores.shape
            
            # Properly extract min/max values from xarray coordinates
            lat_min = float(dataset.coords['lat'].min().values)
            lat_max = float(dataset.coords['lat'].max().values)
            lon_min = float(dataset.coords['lon'].min().values)
            lon_max = float(dataset.coords['lon'].max().values)
            
            lats = np.linspace(lat_min, lat_max, n_lat)
            lons = np.linspace(lon_min, lon_max, n_lon)
            
            # Create meshgrid of coordinates
            lats_grid, lons_grid = np.meshgrid(lats, lons, indexing='ij')
            
            # Flatten arrays
            lats_flat = lats_grid.flatten()
            lons_flat = lons_grid.flatten()
            scores_flat = priority_scores.flatten()
            
            # Create DataFrame
            df = pd.DataFrame({
                'Latitude': lats_flat,
                'Longitude': lons_flat,
                'Priority_Score': scores_flat
            })
            
            # Get top 15 areas (20 can be too many for display)
            df = df.nlargest(15, 'Priority_Score')
            
            # Create location labels with Bengaluru context
            df['Location'] = df.apply(
                lambda x: f"Area {df.index.get_loc(x.name) + 1}: ({x['Latitude']:.2f}, {x['Longitude']:.2f})", 
                axis=1
            )
            
            # Choose color scale based on parameter
            if parameter == 'NDVI':
                color_scale = 'RdYlGn'
            elif parameter == 'LST':
                color_scale = 'RdBu_r'
            elif parameter == 'Precipitation':
                color_scale = 'Blues'
            elif parameter == 'Biomass':
                color_scale = 'Greens'
            else:
                color_scale = 'Viridis'
            
            # Create horizontal bar chart with improved styling
            fig = px.bar(
                df,
                x='Priority_Score',
                y='Location',
                title=f'Top Priority Areas for {parameter}',
                labels={'Priority_Score': 'Priority Score', 'Location': 'Location'},
                color='Priority_Score',
                color_continuous_scale=color_scale,
                height=350
            )
            
            # Improve layout
            fig.update_layout(
                margin=dict(l=50, r=50, t=50, b=50),
                yaxis=dict(autorange="reversed")  # Highest score at top
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            # Fallback visualization
            print(f"Error creating priority chart: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Could not generate priority chart for {parameter}.<br>Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(
                title=f'Priority Areas for {parameter}',
                height=350,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def _create_multi_parameter_chart(self, primary_param: str) -> str:
        """Create a visualization showing relationships between multiple parameters."""
        try:
            # Get all available parameters
            parameters = list(self.predictions.keys())
            
            if len(parameters) < 2:
                # If only one parameter is available, create a heatmap of that parameter
                pred = self.predictions[primary_param]
                
                # Ensure we have enough data
                if pred.shape[0] == 0:
                    raise ValueError("No prediction data available")
                    
                # Calculate the mean over time
                heatmap_data = np.mean(pred, axis=0)
                
                # Create a better color scale based on parameter
                if primary_param == 'NDVI':
                    color_scale = 'RdYlGn'
                elif primary_param == 'LST':
                    color_scale = 'RdBu_r'
                elif primary_param == 'Precipitation':
                    color_scale = 'Blues'
                elif primary_param == 'Biomass':
                    color_scale = 'Greens'
                else:
                    color_scale = 'Viridis'
                
                fig = px.imshow(
                    heatmap_data,
                    title=f'{primary_param} Spatial Distribution',
                    labels=dict(x='Longitude', y='Latitude', color=primary_param),
                    color_continuous_scale=color_scale,
                    height=350,
                    aspect='equal'
                )
            else:
                # If multiple parameters are available, create a correlation heatmap
                correlation_data = {}
                
                # Calculate mean values for each parameter
                for param in parameters:
                    pred = self.predictions[param]
                    if pred.shape[0] > 0:  # Ensure we have data
                        correlation_data[param] = np.mean(pred, axis=(1, 2))  # Average over spatial dimensions
                
                # Create correlation matrix
                df = pd.DataFrame(correlation_data)
                corr_matrix = df.corr()
                
                # Create heatmap with improved styling
                fig = px.imshow(
                    corr_matrix,
                    title='Parameter Correlation Matrix',
                    labels=dict(x='Parameter', y='Parameter', color='Correlation'),
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    height=350,
                    text_auto=True  # Show correlation values
                )
                
                # Add annotations with correlation values
                for i in range(len(corr_matrix.index)):
                    for j in range(len(corr_matrix.columns)):
                        fig.add_annotation(
                            x=j, y=i,
                            text=f"{corr_matrix.iloc[i, j]:.2f}",
                            showarrow=False,
                            font=dict(color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                        )
            
            # Improve layout
            fig.update_layout(
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            # Fallback visualization
            print(f"Error creating multi-parameter chart: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Could not generate multi-parameter visualization.<br>Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(
                title='Multi-parameter Analysis',
                height=350,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def _create_summary(self, parameter: str):
        """Create the summary statistics."""
        dataset = self.data[parameter]
        pred = self.predictions[parameter]
        
        current_avg = float(dataset[parameter].mean().values)
        predicted_avg = float(np.mean(pred))
        
        # Handle division by zero and format change percentage
        if abs(current_avg) > 1e-6:  # Avoid division by near-zero
            change_percent = ((predicted_avg - current_avg) / abs(current_avg)) * 100
            change_percent = np.clip(change_percent, -1000, 1000)  # Limit extreme values
        else:
            change_percent = 0.0
            
        # Format numbers to be human readable
        return {
            'current_avg': f"{current_avg:.2f}",
            'predicted_avg': f"{predicted_avg:.2f}",
            'change_percent': f"{change_percent:.1f}%",
            'trend': 'Improving' if change_percent > 0 else 'Degrading'
        }

    def _reduce_spatial_resolution(self, data: np.ndarray, factor: int = 8) -> np.ndarray:
        """Reduce spatial resolution by averaging over blocks."""
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

    def run(self, debug: bool = False, port: int = 5000):
        """Run the web server."""
        print(f"\n=== Web Interface Started ===")
        print(f"Access the dashboard at: http://localhost:{port}")
        print(f"Available parameters: {list(self.data.keys())}")
        print(f"Press Ctrl+C to stop the server\n")
        self.app.run(debug=debug, port=port) 