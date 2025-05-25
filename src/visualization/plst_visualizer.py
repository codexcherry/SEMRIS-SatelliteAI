"""
Advanced visualization module for PLST analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PLSTVisualizer:
    """
    Visualization tools for PLST analysis results.
    Provides multiple visualization methods for risk assessment and component analysis.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.color_schemes = {
            'risk': ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'],
            'temperature': ['#313695', '#4575b4', '#74add1', '#abd9e9', '#fee090', '#fdae61', '#f46d43', '#d73027'],
            'vegetation': ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63'],
            'drought': ['#730000', '#E60000', '#FFAA00', '#FCD37F', '#FFFF00', '#FFFFFF', '#00FF00', '#005C00']
        }
    
    def plot_risk_map(
        self,
        risk_scores: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        title: str = "Environmental Risk Assessment"
    ) -> folium.Map:
        """
        Create an interactive risk map using Folium.
        
        Args:
            risk_scores: 2D array of risk scores
            lat: Latitude coordinates
            lon: Longitude coordinates
            title: Map title
            
        Returns:
            Folium map object
        """
        # Create base map
        center_lat = np.mean(lat)
        center_lon = np.mean(lon)
        risk_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='cartodbpositron'
        )
        
        # Add risk heatmap
        gradient = {
            0.2: '#2ecc71',  # Low risk - green
            0.4: '#f1c40f',  # Medium risk - yellow
            0.6: '#e67e22',  # High risk - orange
            0.8: '#e74c3c'   # Severe risk - red
        }
        
        folium.plugins.HeatMap(
            data=list(zip(lat.flatten(), lon.flatten(), risk_scores.flatten())),
            gradient=gradient,
            min_opacity=0.5,
            radius=15,
            blur=10,
            max_zoom=13
        ).add_to(risk_map)
        
        # Add title
        title_html = f'<h3 align="center">{title}</h3>'
        risk_map.get_root().html.add_child(folium.Element(title_html))
        
        return risk_map
    
    def plot_component_analysis(
        self,
        component_indices: Dict[str, np.ndarray],
        timestamps: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a comprehensive multi-panel analysis plot.
        
        Args:
            component_indices: Dictionary of component analysis results
            timestamps: List of timestamp strings
            save_path: Optional path to save the plot
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'LST Anomalies',
                'Vegetation Stress',
                'Drought Index',
                'Extreme Events'
            )
        )
        
        # Plot LST anomalies
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=component_indices['lst_anomalies'],
                mode='lines+markers',
                name='LST Anomalies',
                line=dict(color='#e74c3c')
            ),
            row=1, col=1
        )
        
        # Plot vegetation stress
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=component_indices['vegetation_stress'],
                mode='lines+markers',
                name='Vegetation Stress',
                line=dict(color='#2ecc71')
            ),
            row=1, col=2
        )
        
        # Plot drought index
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=component_indices['drought_index'],
                mode='lines+markers',
                name='Drought Index',
                line=dict(color='#3498db')
            ),
            row=2, col=1
        )
        
        # Plot extreme events
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=component_indices['extreme_lst_events'].astype(int),
                name='Extreme Events',
                marker_color='#9b59b6'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text="Component Analysis Dashboard",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_temporal_analysis(
        self,
        risk_scores: np.ndarray,
        timestamps: List[str],
        window_size: int = 30
    ) -> go.Figure:
        """
        Create temporal analysis plot with trends and forecasting.
        
        Args:
            risk_scores: Array of risk scores
            timestamps: List of timestamp strings
            window_size: Rolling window size for trend analysis
            
        Returns:
            Plotly figure object
        """
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'risk_score': risk_scores
        })
        
        # Calculate rolling statistics
        df['rolling_mean'] = df['risk_score'].rolling(window=window_size).mean()
        df['rolling_std'] = df['risk_score'].rolling(window=window_size).std()
        
        # Create temporal analysis plot
        fig = go.Figure()
        
        # Add raw risk scores
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['risk_score'],
                mode='lines',
                name='Risk Score',
                line=dict(color='#3498db', width=1)
            )
        )
        
        # Add rolling mean
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['rolling_mean'],
                mode='lines',
                name=f'{window_size}-day Moving Average',
                line=dict(color='#e74c3c', width=2)
            )
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'].tolist() + df['timestamp'].tolist()[::-1],
                y=(df['rolling_mean'] + 2*df['rolling_std']).tolist() + 
                  (df['rolling_mean'] - 2*df['rolling_std']).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(231, 76, 60, 0.2)',
                line=dict(color='rgba(231, 76, 60, 0)'),
                name='95% Confidence Interval'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Temporal Risk Analysis with Trend',
            xaxis_title='Date',
            yaxis_title='Risk Score',
            height=600,
            width=1000,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def generate_report_figures(
        self,
        risk_data: Dict[str, np.ndarray],
        metadata: Dict,
        save_dir: str
    ) -> List[str]:
        """
        Generate a complete set of figures for the analysis report.
        
        Args:
            risk_data: Dictionary containing risk analysis data
            metadata: Analysis metadata
            save_dir: Directory to save figures
            
        Returns:
            List of saved figure paths
        """
        saved_figures = []
        
        # Risk distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=risk_data['risk_scores'].flatten(),
            bins=50,
            kde=True,
            color='#3498db'
        )
        plt.title('Risk Score Distribution')
        plt.xlabel('Risk Score')
        plt.ylabel('Frequency')
        
        dist_path = f"{save_dir}/risk_distribution.png"
        plt.savefig(dist_path)
        plt.close()
        saved_figures.append(dist_path)
        
        # Component correlation matrix
        components = pd.DataFrame({
            'LST': risk_data['component_indices']['lst_anomalies'].flatten(),
            'Vegetation': risk_data['component_indices']['vegetation_stress'].flatten(),
            'Drought': risk_data['component_indices']['drought_index'].flatten()
        })
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            components.corr(),
            annot=True,
            cmap='RdYlBu_r',
            center=0
        )
        plt.title('Component Correlation Matrix')
        
        corr_path = f"{save_dir}/component_correlations.png"
        plt.savefig(corr_path)
        plt.close()
        saved_figures.append(corr_path)
        
        return saved_figures
    
    def generate_visualization_data(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualization data from analysis report.
        
        Args:
            report: Analysis report from PLST Agent
            
        Returns:
            Dictionary of visualization data
        """
        viz_data = {
            'time_series': {},
            'spatial_data': {},
            'risk_scores': {},
            'anomalies': []
        }
        
        # Extract analysis results
        analysis = report.get('analysis', {})
        
        # Process risk scores
        if 'risk_scores' in analysis:
            risk_scores = analysis['risk_scores']
            if isinstance(risk_scores, list):
                viz_data['risk_scores'] = {
                    'values': risk_scores,
                    'mean': float(np.mean(risk_scores)),
                    'max': float(np.max(risk_scores)),
                    'min': float(np.min(risk_scores))
                }
        
        # Process temporal patterns
        if 'temporal_patterns' in analysis:
            temporal = analysis['temporal_patterns']
            if temporal:
                viz_data['time_series'] = {
                    'trends': temporal.get('trends', {}),
                    'time_range': temporal.get('time_range', {})
                }
        
        # Process anomalies
        if 'anomaly_scores' in analysis:
            anomaly_scores = analysis['anomaly_scores']
            if isinstance(anomaly_scores, list):
                anomalies = []
                for i, score in enumerate(anomaly_scores):
                    if score < -0.5:  # Significant anomaly
                        anomalies.append({
                            'index': i,
                            'score': score,
                            'severity': 'high' if score < -0.8 else 'medium'
                        })
                viz_data['anomalies'] = anomalies
                
                # Create anomaly distribution data
                viz_data['anomaly_distribution'] = {
                    'values': anomaly_scores,
                    'mean': float(np.mean(anomaly_scores)),
                    'max': float(np.max(anomaly_scores)),
                    'min': float(np.min(anomaly_scores))
                }
        
        # Generate vegetation health index data
        if 'component_indices' in analysis:
            component_indices = analysis['component_indices']
            if 'vegetation_stress' in component_indices:
                veg_stress = component_indices['vegetation_stress']
                if isinstance(veg_stress, np.ndarray):
                    # Convert to vegetation health (inverse of stress)
                    veg_health = 1.0 - veg_stress
                    viz_data['vegetation_health_index'] = {
                        'values': veg_health.tolist(),
                        'mean': float(np.mean(veg_health)),
                        'max': float(np.max(veg_health)),
                        'min': float(np.min(veg_health))
                    }
            
            # Generate water resource status data
            if 'drought_index' in component_indices:
                drought_index = component_indices['drought_index']
                if isinstance(drought_index, np.ndarray):
                    # Convert drought index to water resource status (inverse relationship)
                    water_status = 1.0 - (drought_index + 1.0) / 2.0  # Normalize to 0-1 range
                    water_status = np.clip(water_status, 0, 1)  # Ensure values are in 0-1 range
                    viz_data['water_resource_status'] = {
                        'values': water_status.tolist(),
                        'mean': float(np.mean(water_status)),
                        'max': float(np.max(water_status)),
                        'min': float(np.min(water_status))
                    }
        
        # Generate color maps for spatial visualization
        viz_data['color_maps'] = self._generate_color_maps()
        
        return viz_data
    
    def _generate_color_maps(self) -> Dict[str, List[str]]:
        """
        Generate color maps for different visualization types.
        
        Returns:
            Dictionary of color maps
        """
        return {
            'risk': ['#00ff00', '#ffff00', '#ff0000'],  # Green to yellow to red
            'temperature': ['#0000ff', '#ffffff', '#ff0000'],  # Blue to white to red
            'vegetation': ['#8B4513', '#FFFF00', '#006400'],  # Brown to yellow to green
            'anomaly': ['#00ff00', '#ff0000']  # Green to red
        } 