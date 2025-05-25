"""
SEMRIS - Satellite-based Environmental Monitoring and Restoration Intelligence System
Main application entry point with advanced PLST analysis and AI agent integration.
"""

import os
import yaml
from datetime import datetime, timedelta
import numpy as np
from src.data_retrieval.nasa_api import NASADataRetriever
from src.preprocessing.data_cleaner import DataCleaner
from src.modeling.plst_analyzer import PLSTAnalyzer
from src.visualization.plst_visualizer import PLSTVisualizer
from src.region_selection.coordinate_handler import CoordinateHandler
from src.ai_agent.plst_agent import PLSTAgent
from flask import Flask, render_template, jsonify, request
import pandas as pd

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

class SEMRISApp:
    def __init__(self):
        # Load configuration
        with open('config/config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_retriever = NASADataRetriever(self.config)
        self.data_cleaner = DataCleaner(self.config)
        self.plst_analyzer = PLSTAnalyzer(self.config)
        self.visualizer = PLSTVisualizer(self.config)
        self.coordinate_handler = CoordinateHandler(self.config)
        
        # Initialize AI agent
        self.agent = PLSTAgent(self.config)
        
        # Select initial region (Bangalore as default)
        self.current_region = {
            'lat': 12.9716,
            'lon': 77.5946,
            'radius': 50  # km
        }
        print(f"Selecting region: {self.current_region['lat']}, {self.current_region['lon']} with radius {self.current_region['radius']}km")
        
        # Pre-fetch data on startup
        self.cached_report = None
        self.cached_viz_data = None
        self.prefetch_data()
        
    def prefetch_data(self):
        """
        Pre-fetch data on startup to avoid delays when accessing the dashboard
        """
        print("Pre-fetching environmental data...")
        # Get data for the past year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Fetch raw data
        raw_data = self.data_retriever.get_data(
            self.current_region['lat'],
            self.current_region['lon'],
            self.current_region['radius'],
            start_date,
            end_date
        )
        
        print("Preprocessing data...")
        # Clean and preprocess data
        processed_data = self.data_cleaner.clean_data(raw_data)
        
        print("Performing AI-enhanced PLST analysis...")
        # Prepare analysis input
        analysis_input = {
            'data': processed_data,
            'location': self.current_region,
            'timestamp': datetime.now().isoformat(),
            'query_context': {
                'location': self.current_region,
                'timestamp': datetime.now().isoformat(),
                'parameters': ['LST', 'NDVI', 'Precipitation', 'SoilMoisture']
            }
        }
        
        # Perform AI-enhanced analysis
        self.cached_report = self.agent.analyze(
            analysis_input,
            temporal_filter=365  # Consider last year's data
        )
        
        # Generate visualization data
        self.cached_viz_data = self.visualizer.generate_visualization_data(self.cached_report)
        print("Data pre-fetching complete!")
    
    def fetch_and_analyze_data(self, query_context=None):
        """
        Fetch and analyze environmental data for the current region.
        """
        print("Retrieving environmental data...")
        # Get data for the past year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Fetch raw data
        raw_data = self.data_retriever.get_data(
            self.current_region['lat'],
            self.current_region['lon'],
            self.current_region['radius'],
            start_date,
            end_date
        )
        
        print("Preprocessing data...")
        # Clean and preprocess data
        processed_data = self.data_cleaner.clean_data(raw_data)
        
        print("Performing AI-enhanced PLST analysis...")
        # Prepare analysis input
        analysis_input = {
            'data': processed_data,
            'location': self.current_region,
            'timestamp': datetime.now().isoformat(),
            'query_context': query_context or {}
        }
        
        # Perform AI-enhanced analysis
        report = self.agent.analyze(
            analysis_input,
            temporal_filter=365  # Consider last year's data
        )
        
        return report
    
    def get_visualization_data(self, report):
        """
        Generate visualization data from analysis report.
        """
        return self.visualizer.generate_visualization_data(report)
    
    def update_region(self, lat, lon, radius=None):
        """
        Update the current region of interest.
        """
        self.current_region['lat'] = float(lat)
        self.current_region['lon'] = float(lon)
        if radius:
            self.current_region['radius'] = float(radius)

@app.route('/')
def index():
    """
    Render the main dashboard.
    """
    # Convert cached data to JSON-serializable format
    report_json = convert_numpy_to_python(app_instance.cached_report) if app_instance.cached_report else {}
    viz_data_json = convert_numpy_to_python(app_instance.cached_viz_data) if app_instance.cached_viz_data else {}
    
    return render_template(
        'index.html',
        title='SEMRIS - SATELLITEAI',
        region=app_instance.current_region,
        report=report_json,
        visualization=viz_data_json,
        prefetched=True
    )

def convert_numpy_to_python(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_to_python(i) for i in obj]
    else:
        return obj

@app.route('/api/analyze')
def analyze():
    """
    Return analysis for the current region using cached data.
    """
    # Convert cached data to JSON-serializable format
    report_json = convert_numpy_to_python(app_instance.cached_report) if app_instance.cached_report else {}
    viz_data_json = convert_numpy_to_python(app_instance.cached_viz_data) if app_instance.cached_viz_data else {}
    
    return jsonify({
        'status': 'success',
        'data': {
            'report': report_json,
            'visualization': viz_data_json
        }
    })

@app.route('/api/agent/state')
def get_agent_state():
    """
    Get current AI agent state.
    """
    return jsonify({
        'status': 'success',
        'data': app_instance.agent.get_state()
    })

def main():
    """
    Main application entry point.
    """
    print("Starting web interface...")
    global app_instance
    app_instance = SEMRISApp()
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main() 