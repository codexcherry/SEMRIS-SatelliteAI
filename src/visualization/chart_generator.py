import plotly.graph_objects as go
import plotly.express as px
import folium
from folium import plugins
import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Union
import pandas as pd

class ChartGenerator:
    """Generates interactive visualizations for environmental data."""
    
    def __init__(self):
        """Initialize the ChartGenerator."""
        self.color_schemes = {
            'ndvi': ['#8b0000', '#ffffff', '#006400'],  # Red to Green
            'lst': ['#0000ff', '#ffffff', '#ff0000'],   # Blue to Red
            'precipitation': ['#ffffff', '#0000ff']     # White to Blue
        }
    
    def create_time_series(self,
                          data: xr.Dataset,
                          parameter: str,
                          location: Optional[Dict[str, float]] = None) -> go.Figure:
        """
        Create an interactive time series plot.
        
        Args:
            data (xr.Dataset): Input dataset
            parameter (str): Parameter to plot
            location (Dict[str, float]): Optional location to plot (lat, lon)
            
        Returns:
            go.Figure: Interactive time series plot
        """
        if location:
            # Extract time series for specific location
            ts_data = data[parameter].sel(
                lat=location['lat'],
                lon=location['lon'],
                method='nearest'
            )
        else:
            # Use mean across all locations
            ts_data = data[parameter].mean(dim=['lat', 'lon'])
        
        # Create figure
        fig = go.Figure()
        
        # Add time series
        fig.add_trace(go.Scatter(
            x=ts_data.time,
            y=ts_data.values,
            mode='lines',
            name=parameter,
            line=dict(color=self._get_color(parameter))
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{parameter} Time Series',
            xaxis_title='Time',
            yaxis_title=parameter,
            hovermode='x unified'
        )
        
        return fig
    
    def create_spatial_heatmap(self,
                              data: xr.Dataset,
                              parameter: str,
                              time: Optional[str] = None) -> folium.Map:
        """
        Create an interactive spatial heatmap.
        
        Args:
            data (xr.Dataset): Input dataset
            parameter (str): Parameter to plot
            time (str): Optional specific time to plot
            
        Returns:
            folium.Map: Interactive map with heatmap overlay
        """
        # Select time if specified
        if time:
            plot_data = data[parameter].sel(time=time)
        else:
            plot_data = data[parameter].isel(time=-1)  # Use most recent time
        
        # Create base map
        center_lat = float(plot_data.lat.mean())
        center_lon = float(plot_data.lon.mean())
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
        
        # Create heatmap data
        heat_data = []
        for lat in plot_data.lat:
            for lon in plot_data.lon:
                value = float(plot_data.sel(lat=lat, lon=lon))
                if not np.isnan(value):
                    heat_data.append([float(lat), float(lon), value])
        
        # Add heatmap layer
        plugins.HeatMap(heat_data).add_to(m)
        
        return m
    
    def create_degradation_hotspot_map(self,
                                     data: xr.Dataset,
                                     parameter: str,
                                     threshold: float) -> folium.Map:
        """
        Create a map highlighting degradation hotspots.
        
        Args:
            data (xr.Dataset): Input dataset
            parameter (str): Parameter to analyze
            threshold (float): Threshold for degradation
            
        Returns:
            folium.Map: Interactive map with degradation hotspots
        """
        # Calculate degradation (negative trend)
        trend = self._calculate_trend(data[parameter])
        
        # Create base map
        center_lat = float(data.lat.mean())
        center_lon = float(data.lon.mean())
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
        
        # Add degradation hotspots
        for lat in data.lat:
            for lon in data.lon:
                value = float(trend.sel(lat=lat, lon=lon))
                if value < -threshold:
                    folium.CircleMarker(
                        location=[float(lat), float(lon)],
                        radius=5,
                        color='red',
                        fill=True,
                        popup=f'Degradation: {value:.2f}'
                    ).add_to(m)
        
        return m
    
    def create_recovery_trend_plot(self,
                                 data: xr.Dataset,
                                 parameter: str,
                                 window: int = 12) -> go.Figure:
        """
        Create a plot showing recovery trends.
        
        Args:
            data (xr.Dataset): Input dataset
            parameter (str): Parameter to analyze
            window (int): Window size for trend calculation
            
        Returns:
            go.Figure: Interactive recovery trend plot
        """
        # Calculate moving average
        ma = data[parameter].rolling(time=window).mean()
        
        # Create figure
        fig = go.Figure()
        
        # Add original data
        fig.add_trace(go.Scatter(
            x=data.time,
            y=data[parameter].mean(dim=['lat', 'lon']),
            mode='lines',
            name='Original',
            line=dict(color='gray')
        ))
        
        # Add moving average
        fig.add_trace(go.Scatter(
            x=ma.time,
            y=ma.mean(dim=['lat', 'lon']),
            mode='lines',
            name='Trend',
            line=dict(color=self._get_color(parameter))
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{parameter} Recovery Trend',
            xaxis_title='Time',
            yaxis_title=parameter,
            hovermode='x unified'
        )
        
        return fig
    
    def _calculate_trend(self, data: xr.DataArray) -> xr.DataArray:
        """
        Calculate linear trend over time.
        
        Args:
            data (xr.DataArray): Input data
            
        Returns:
            xr.DataArray: Trend values
        """
        # Reshape data for trend calculation
        time = np.arange(len(data.time))
        trend = np.zeros((len(data.lat), len(data.lon)))
        
        for i, lat in enumerate(data.lat):
            for j, lon in enumerate(data.lon):
                series = data.sel(lat=lat, lon=lon).values
                if not np.all(np.isnan(series)):
                    # Calculate linear trend
                    z = np.polyfit(time, series, 1)
                    trend[i, j] = z[0]  # Slope
        
        return xr.DataArray(
            trend,
            coords={'lat': data.lat, 'lon': data.lon},
            dims=['lat', 'lon']
        )
    
    def _get_color(self, parameter: str) -> str:
        """
        Get color for a parameter.
        
        Args:
            parameter (str): Parameter name
            
        Returns:
            str: Color code
        """
        return self.color_schemes.get(parameter.lower(), ['#000000'])[-1] 