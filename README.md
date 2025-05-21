# SEMRIS (Satellite-based Environmental Monitoring and Restoration Intelligence System)

A comprehensive system for monitoring and analyzing environmental changes using satellite data and machine learning, with a focus on Bengaluru region.

## Project Overview

SEMRIS is designed to provide actionable insights for environmental monitoring and restoration efforts through the analysis of satellite data. The system processes various environmental parameters to predict vegetation health, biomass levels, and identify potential degradation areas, with a specific focus on the Bengaluru metropolitan region.

## Key Features

### 1. Region Selection
- Geographic coordinate-based region selection
- Focus on Bengaluru region (default coordinates: 12.97°N, 77.59°E)
- Configurable radius for area of interest

### 2. Data Retrieval
- NASA Earthdata API integration
- Multiple environmental parameters support
- Automated data fetching for specified time periods

### 3. Data Preprocessing
- Automated data cleaning and validation
- Time series normalization
- Spatial data processing
- Quality assessment and validation

### 4. Predictive Modeling
- RNN-based architecture using PyTorch
- Multi-parameter forecasting
- Configurable model parameters
- Automated training pipeline

### 5. Web Interface
- Interactive dashboard
- Real-time data visualization
- Prediction results display
- User-friendly interface

## Project Structure

```
SEMRIS-SatelliteAI/
├── src/
│   ├── region_selection/
│   │   └── coordinate_handler.py
│   ├── data_retrieval/
│   │   └── nasa_api.py
│   ├── preprocessing/
│   │   └── data_cleaner.py
│   ├── modeling/
│   │   └── rnn_model.py
│   └── insights/
│       └── web_interface.py
├── config/
│   └── config.yaml
├── templates/
├── static/
├── tests/
├── logs/
├── reports/
├── requirements.txt
└── main.py
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SEMRIS-SatelliteAI.git
cd SEMRIS-SatelliteAI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the application:
- Review and modify `config/config.yaml` for your specific needs
- Set up any required API keys

## Usage

1. Run the application:
```bash
python main.py
```

The application will:
- Select the Bengaluru region (configurable)
- Retrieve environmental data from NASA
- Preprocess the data
- Train predictive models
- Launch a web interface on port 5000

2. Access the web interface:
- Open your browser and navigate to `http://localhost:5000`
- View environmental data and predictions
- Interact with the visualization tools

## Dependencies

Key dependencies include:
- Flask (Web interface)
- PyTorch (Deep learning)
- NASA API (Data retrieval)
- Pandas & NumPy (Data processing)
- Plotly (Visualization)

For a complete list of dependencies, see `requirements.txt`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NASA Earthdata API for satellite data
- PyTorch for deep learning capabilities
- Flask for web interface
- All other open-source contributors 