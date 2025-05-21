# SEMRIS (Satellite-based Environmental Monitoring and Restoration Intelligence System)

A comprehensive system for monitoring and analyzing environmental changes using satellite data and machine learning.

## Project Overview

SEMRIS is designed to provide actionable insights for environmental monitoring and restoration efforts through the analysis of satellite data. The system processes various environmental parameters to predict vegetation health, biomass levels, and identify potential degradation areas.

## Workflow Stages

### 1. Region Selection
- Geographic coordinate input
- Shapefile support
- Area of interest definition
- Boundary validation

### 2. Data Retrieval
- NASA Earthdata API integration
- Google Earth Engine data access
- Environmental parameters:
  - NDVI (Normalized Difference Vegetation Index)
  - Biomass measurements
  - Land Surface Temperature
  - Precipitation data
  - Land Cover Classification

### 3. Data Preprocessing
- Data cleaning and validation
- Time series normalization
- Spatial interpolation
- Temporal resampling
- Data quality assessment

### 4. Predictive Modeling
- RNN-based architecture using PyTorch
- Temporal pattern recognition
- Multi-parameter forecasting
- Model validation and evaluation
- Prediction confidence scoring

### 5. Visualization
- Interactive time series charts
- Spatial data overlays
- Heat maps for degradation hotspots
- Recovery trend analysis
- Custom visualization tools

### 6. Insights Delivery
- Web-based dashboard
- Automated reporting
- Decision support tools
- Export capabilities
- API endpoints for integration

## Project Structure

```
SEMRIS-SatelliteAI/
├── src/
│   ├── region_selection/
│   │   ├── __init__.py
│   │   ├── coordinate_handler.py
│   │   └── shapefile_processor.py
│   ├── data_retrieval/
│   │   ├── __init__.py
│   │   ├── nasa_api.py
│   │   └── gee_connector.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_cleaner.py
│   │   └── time_series_processor.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── rnn_model.py
│   │   └── model_trainer.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── chart_generator.py
│   │   └── spatial_visualizer.py
│   └── insights/
│       ├── __init__.py
│       ├── dashboard.py
│       └── report_generator.py
├── config/
│   ├── config.yaml
│   └── api_keys.yaml
├── tests/
│   └── __init__.py
├── requirements.txt
└── README.md
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SEMRIS-SatelliteAI.git
cd SEMRIS-SatelliteAI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys:
- Copy `config/api_keys.yaml.example` to `config/api_keys.yaml`
- Add your NASA Earthdata API key and other required credentials

## Usage

1. Region Selection:
```python
from src.region_selection.coordinate_handler import CoordinateHandler

handler = CoordinateHandler()
region = handler.select_region(lat=45.0, lon=-120.0, radius_km=50)
```

2. Data Retrieval:
```python
from src.data_retrieval.nasa_api import NASADataRetriever

retriever = NASADataRetriever()
data = retriever.get_environmental_data(region, parameters=['NDVI', 'LST'])
```

3. Run Analysis:
```python
from src.modeling.model_trainer import ModelTrainer

trainer = ModelTrainer()
model = trainer.train_model(data)
predictions = model.predict_future_trends()
```

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NASA Earthdata API
- Google Earth Engine
- PyTorch
- Various open-source geospatial libraries 