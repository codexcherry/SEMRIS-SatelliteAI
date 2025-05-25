import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import xarray as xr
from typing import Dict, Any, Tuple
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import base64

class ReportGenerator:
    def __init__(self, data: Dict[str, xr.Dataset], predictions: Dict[str, np.ndarray]):
        """
        Initialize the report generator.
        
        Args:
            data: Dictionary of xarray Datasets containing historical data
            predictions: Dictionary of numpy arrays containing model predictions
        """
        self.data = data
        self.predictions = predictions
        self.output_dir = "reports"
        os.makedirs(self.output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=18,
            spaceAfter=20
        )
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12
        )
        
    def _resample_data(self, dataset: xr.Dataset, parameter: str, freq: str = 'ME') -> xr.Dataset:
        """Resample data to reduce size while maintaining trends."""
        return dataset.resample(time=freq).mean()
        
    def _reduce_spatial_resolution(self, data: np.ndarray, factor: int = 8) -> np.ndarray:
        """Reduce spatial resolution by averaging over blocks. Always returns a valid reduced array."""
        n_time, n_lat, n_lon = data.shape
        if n_lat < factor or n_lon < factor:
            print(f"[WARN] Data too small for reduction factor {factor}, using mean over spatial axes.")
            return data.mean(axis=(1,2), keepdims=True)  # shape (n_time, 1, 1)
        new_lat = n_lat // factor
        new_lon = n_lon // factor
        cropped_data = data[:, :new_lat*factor, :new_lon*factor]
        try:
            reshaped = cropped_data.reshape(n_time, new_lat, factor, new_lon, factor)
            return np.mean(reshaped, axis=(2, 4))
        except Exception as e:
            print(f"[WARN] Could not reshape for reduction: {e}. Using mean over spatial axes.")
            return data.mean(axis=(1,2), keepdims=True)
        
    def _create_plot_image(self, fig) -> bytes:
        """Convert plotly figure to image bytes."""
        img_bytes = fig.to_image(format="png", width=800, height=400)
        return img_bytes

    def generate_comprehensive_report(self, parameter: str = 'NDVI') -> str:
        """Generate a comprehensive PDF report with all visualizations and analysis."""
        print("Generating comprehensive environmental report...")
        
        # Create PDF document
        output_path = f"{self.output_dir}/environmental_report.pdf"
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []

        # Title
        story.append(Paragraph("Environmental Forecast Report", self.title_style))
        story.append(Spacer(1, 20))

        # 1. Region Summary
        story.append(Paragraph("Region Summary", self.heading_style))
        dataset = self.data[parameter]
        pred = self.predictions[parameter]
        
        # Calculate statistics
        current_avg = float(dataset[parameter].mean().values)
        predicted_avg = float(np.mean(pred))
        change_percent = ((predicted_avg - current_avg) / current_avg) * 100
        
        summary_text = f"""
        Current Average: {current_avg:.2f}
        Predicted Average: {predicted_avg:.2f}
        Change: {change_percent:.1f}%
        Trend: {'Improving' if change_percent > 0 else 'Degrading'}
        """
        story.append(Paragraph(summary_text, self.normal_style))
        story.append(Spacer(1, 20))

        # 2. Trend Visualization
        story.append(Paragraph("Historical and Predicted Trends", self.heading_style))
        dataset = self._resample_data(dataset, parameter, freq='ME')
        pred = pred[::60]  # Take every 60th prediction
        
        hist_data = dataset[parameter].mean(dim=['lat', 'lon']).values
        hist_dates = pd.to_datetime(dataset.time.values)
        pred_data = pred.mean(axis=(1, 2))
        last_date = hist_dates[-1]
        pred_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=len(pred_data),
            freq='2ME'
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_dates, y=hist_data, name='Historical', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=pred_dates, y=pred_data, name='Predicted', line=dict(color='red', dash='dash')))
        fig.update_layout(
            title=f'{parameter} Historical and Predicted Trends',
            xaxis_title='Date',
            yaxis_title=parameter,
            template='plotly_white'
        )
        
        img_bytes = self._create_plot_image(fig)
        img = Image(io.BytesIO(img_bytes), width=6*inch, height=3*inch)
        story.append(img)
        story.append(Spacer(1, 20))

        # 3. Risk Map
        story.append(Paragraph("Degradation Risk Map", self.heading_style))
        pred = self._reduce_spatial_resolution(pred)
        risk_map = np.mean(pred, axis=0)
        risk_map = (risk_map - np.min(risk_map)) / (np.max(risk_map) - np.min(risk_map))
        
        fig = px.imshow(
            risk_map,
            color_continuous_scale='RdYlGn_r',
            title='Degradation Risk Map',
            labels=dict(x='Longitude', y='Latitude', color='Risk Level')
        )
        
        img_bytes = self._create_plot_image(fig)
        img = Image(io.BytesIO(img_bytes), width=6*inch, height=3*inch)
        story.append(img)
        story.append(Spacer(1, 20))

        # 4. Restoration Priority
        story.append(Paragraph("Restoration Priority Areas", self.heading_style))
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
        
        fig = px.bar(
            df,
            x='Priority_Score',
            y='Latitude',
            title='Top 20 Areas by Restoration Priority',
            labels={'Priority_Score': 'Restoration Priority Score', 'Latitude': 'Location'},
            color='Priority_Score',
            color_continuous_scale='RdYlGn_r'
        )
        
        img_bytes = self._create_plot_image(fig)
        img = Image(io.BytesIO(img_bytes), width=6*inch, height=3*inch)
        story.append(img)

        # Build PDF
        doc.build(story)
        print("Comprehensive report generated successfully")
        return output_path
        
    def generate_vegetation_forecast(self, parameter: str = 'NDVI') -> str:
        """Generate vegetation forecast CSV with geo-tagged data."""
        print("Generating vegetation forecast...")
        if parameter not in self.predictions:
            raise ValueError(f"No predictions available for {parameter}")
            
        pred = self.predictions[parameter]
        dataset = self.data[parameter]
        
        # Resample to monthly data and reduce spatial resolution
        dataset = self._resample_data(dataset, parameter, freq='ME')
        pred = pred[::60]  # Take every 60th prediction (approximately bi-monthly)
        pred = self._reduce_spatial_resolution(pred)
        
        # Create time index for predictions
        last_date = pd.Timestamp(dataset.time.values[-1])
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=len(pred),
            freq='2ME'  # Bi-monthly end of month
        )
        
        # Create DataFrame with coordinates (reduced resolution)
        lats = dataset.coords['lat'].values[::8]  # Take every 8th point
        lons = dataset.coords['lon'].values[::8]
        
        # Vectorized operation to create forecast data
        dates = np.repeat(future_dates, len(lats) * len(lons))
        lats_grid = np.tile(np.repeat(lats, len(lons)), len(future_dates))
        lons_grid = np.tile(np.tile(lons, len(lats)), len(future_dates))
        values = pred.reshape(-1)
        
        df = pd.DataFrame({
            'Date': dates,
            'Latitude': lats_grid,
            'Longitude': lons_grid,
            f'{parameter}_Predicted': values
        })
        
        output_path = f"{self.output_dir}/vegetation_forecast.csv"
        df.to_csv(output_path, index=False)
        print("Vegetation forecast generated successfully")
        return output_path
        
    def generate_trend_visualization(self, parameter: str = 'NDVI') -> str:
        """Generate environmental trend visualization comparing historical and predicted data."""
        print("Generating trend visualization...")
        dataset = self.data[parameter]
        pred = self.predictions[parameter]
        
        # Resample to monthly data
        dataset = self._resample_data(dataset, parameter, freq='ME')
        pred = pred[::60]  # Take every 60th prediction
        
        # Get historical data using optimized operations
        hist_data = dataset[parameter].mean(dim=['lat', 'lon']).values
        hist_dates = pd.to_datetime(dataset.time.values)
        
        # Get predicted data
        pred_data = pred.mean(axis=(1, 2))
        last_date = hist_dates[-1]
        pred_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=len(pred_data),
            freq='2M'
        )
        
        # Create simplified figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_dates, y=hist_data, name='Historical', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=pred_dates, y=pred_data, name='Predicted', line=dict(color='red', dash='dash')))
        
        fig.update_layout(
            title=f'{parameter} Historical and Predicted Trends',
            xaxis_title='Date',
            yaxis_title=parameter,
            template='plotly_white'
        )
        
        output_path = f"{self.output_dir}/environmental_trend_plot.png"
        fig.write_image(output_path, width=800, height=400)  # Reduced image size
        print("Trend visualization generated successfully")
        return output_path
        
    def generate_degradation_risk_map(self, parameter: str = 'NDVI') -> str:
        """Generate degradation risk heatmap."""
        print("Generating degradation risk map...")
        pred = self.predictions[parameter]
        
        # Reduce resolution for faster processing
        pred = self._reduce_spatial_resolution(pred)
        
        # Calculate risk based on predicted values using vectorized operations
        risk_map = np.mean(pred, axis=0)  # Average over time
        risk_map = (risk_map - np.min(risk_map)) / (np.max(risk_map) - np.min(risk_map))
        
        # Create figure with reduced size
        fig = px.imshow(
            risk_map,
            color_continuous_scale='RdYlGn_r',
            title='Degradation Risk Map',
            labels=dict(x='Longitude', y='Latitude', color='Risk Level'),
            width=800,
            height=400
        )
        
        output_path = f"{self.output_dir}/degradation_risk_map.png"
        fig.write_image(output_path)
        print("Degradation risk map generated successfully")
        return output_path
        
    def generate_region_summary(self, parameter: str = 'NDVI') -> str:
        """Generate regional summary report in JSON format."""
        print("Generating region summary...")
        dataset = self.data[parameter]
        pred = self.predictions[parameter]
        
        # Calculate statistics using vectorized operations
        current_avg = float(dataset[parameter].mean().values)
        predicted_avg = float(np.mean(pred))
        change_percent = ((predicted_avg - current_avg) / current_avg) * 100
        
        # Calculate risk map for summary
        risk_map = np.mean(pred, axis=0)
        risk_map = (risk_map - np.min(risk_map)) / (np.max(risk_map) - np.min(risk_map))
        
        summary = {
            'region_info': {
                'latitude_range': [float(dataset.coords['lat'].min()), float(dataset.coords['lat'].max())],
                'longitude_range': [float(dataset.coords['lon'].min()), float(dataset.coords['lon'].max())],
                'area_km2': float(dataset.attrs.get('area_km2', 0))
            },
            'vegetation_health': {
                'current_average': current_avg,
                'predicted_average': predicted_avg,
                'percent_change': change_percent,
                'trend': 'improving' if change_percent > 0 else 'degrading'
            },
            'risk_assessment': {
                'high_risk_areas_percent': float(np.mean(risk_map > 0.7) * 100),
                'moderate_risk_areas_percent': float(np.mean((risk_map > 0.4) & (risk_map <= 0.7)) * 100),
                'low_risk_areas_percent': float(np.mean(risk_map <= 0.4) * 100)
            },
            'model_confidence': {
                'confidence_level': 'High',
                'data_sources': ['Multi-temporal satellite observations', 'Historical vegetation indices'],
                'prediction_horizon': '6 months'
            }
        }
        
        output_path = f"{self.output_dir}/region_summary_report.json"
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print("Region summary generated successfully")
        return output_path
        
    def generate_restoration_priority_chart(self, parameter: str = 'NDVI') -> str:
        """Generate restoration priority chart."""
        print("Generating restoration priority chart...")
        pred = self.predictions[parameter]
        dataset = self.data[parameter]
        
        # Reduce resolution for faster processing
        pred = self._reduce_spatial_resolution(pred)
        
        # Calculate priority scores using vectorized operations
        priority_scores = np.mean(pred, axis=0)
        priority_scores = (priority_scores - np.min(priority_scores)) / (np.max(priority_scores) - np.min(priority_scores))
        
        # Create DataFrame for plotting using vectorized operations
        lats = dataset.coords['lat'].values[::8]  # Take every 8th point
        lons = dataset.coords['lon'].values[::8]
        
        # Vectorized operation to create priority data
        lats_grid = np.repeat(lats, len(lons))
        lons_grid = np.tile(lons, len(lats))
        scores = priority_scores.reshape(-1)
        
        df = pd.DataFrame({
            'Latitude': lats_grid,
            'Longitude': lons_grid,
            'Priority_Score': scores
        })
        
        df = df.nlargest(20, 'Priority_Score')  # Get top 20 areas
        
        fig = px.bar(
            df,
            x='Priority_Score',
            y='Latitude',
            title='Top 20 Areas by Restoration Priority',
            labels={'Priority_Score': 'Restoration Priority Score', 'Latitude': 'Location'},
            color='Priority_Score',
            color_continuous_scale='RdYlGn_r'
        )
        
        output_path = f"{self.output_dir}/restoration_priority_chart.png"
        fig.write_image(output_path, width=800, height=400)  # Reduced image size
        print("Restoration priority chart generated successfully")
        return output_path
        
    def generate_all_reports(self, parameter: str = 'NDVI') -> str:
        """Generate a single comprehensive PDF report."""
        return self.generate_comprehensive_report(parameter) 