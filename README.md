# SEMRIS (Satellite-based Environmental Monitoring and Restoration Intelligence System)

A comprehensive system for monitoring and analyzing environmental changes using satellite data and AI agents, with a focus on the Bengaluru region.

## Project Overview

SEMRIS is designed to provide actionable insights for environmental monitoring and restoration efforts through the analysis of satellite data. The system processes various environmental parameters to predict vegetation health, biomass levels, and identify potential degradation areas, with a specific focus on the Bengaluru metropolitan region.

## Key Features

### 1. AI Agent System
- Environmental Analysis Agent for processing satellite data
- Prediction Agent for forecasting environmental changes
- Coordination Agent for managing multi-agent interactions
- RAG (Retrieval-Augmented Generation) for context-aware analysis
- MCP (Model Context Protocol) for maintaining agent state and memory
- ADK (Agent Development Kit) for agent learning and improvement

### 2. Region Selection
- Geographic coordinate-based region selection
- Focus on Bengaluru region (default coordinates: 12.97°N, 77.59°E)
- Configurable radius for area of interest

### 3. Data Retrieval
- NASA Earthdata API integration
- Multiple environmental parameters support
- Automated data fetching for specified time periods

### 4. Data Processing
- Automated data cleaning and validation
- Time series normalization
- Spatial data processing
- Quality assessment and validation
- Context-aware analysis using RAG

### 5. Predictive Modeling
- RNN-based architecture using PyTorch
- Multi-parameter forecasting
- Configurable model parameters
- Automated training pipeline
- Agent-based learning and adaptation

### 6. Visualization
- Interactive data visualization
- Real-time monitoring dashboards
- Custom plotting utilities
- Geographic data visualization

## Project Structure

The SEMRIS project is organized for clarity, modularity, and scalability. Below is a detailed overview of the directory structure and the purpose of each component:

```
SEMRIS-SatelliteAI/
├── src/
│   ├── ai_agent/                  # Core AI agent components (base, environmental, prediction, coordination agents)
│   ├── agents/                    # Legacy agent components (for backward compatibility or migration)
│   ├── rag/                       # Retrieval Augmented Generation (contextual retrieval and response generation)
│   ├── mcp/                       # Model Context Protocol (context, memory, and state management)
│   ├── adk/                       # Agent Development Kit (learning, behavior, and metrics for agents)
│   ├── visualization/             # Data visualization utilities and dashboards
│   ├── region_selection/          # Tools for selecting and managing geographic regions of interest
│   ├── data_retrieval/            # Modules for fetching and managing external data (e.g., satellite, NASA API)
│   ├── preprocessing/             # Data cleaning, normalization, and preparation utilities
│   ├── modeling/                  # Machine learning models and training scripts
│   └── insights/                  # Analysis and insight generation modules
│
├── config/                        # Configuration files (YAML) for agents, RAG, MCP, and general settings
├── templates/                     # HTML templates for the web interface
├── static/                        # Static assets (CSS, JS, images) for the web interface
├── tests/                         # Unit and integration tests for all modules
├── logs/                          # System and application logs
├── reports/                       # Generated analysis and summary reports
├── venv/                          # Python virtual environment (not tracked in version control)
├── requirements.txt               # Python dependencies
├── main.py                        # Main entry point for running the application
├── README.md                      # Project documentation (this file)
└── project_structure.md           # Detailed project structure reference
```

### Directory and File Descriptions

- **src/ai_agent/**: Contains the main AI agent classes, including base agents and specialized agents for environmental analysis, prediction, and coordination.
- **src/agents/**: Holds legacy or experimental agent code, useful for reference or migration.
- **src/rag/**: Implements Retrieval Augmented Generation, including document storage, retrieval, and response generation for context-aware analysis.
- **src/mcp/**: Manages agent state, memory, and context to ensure consistent and persistent multi-agent operations.
- **src/adk/**: Provides tools for agent learning, behavior adjustment, and performance tracking.
- **src/visualization/**: Contains scripts and modules for visualizing data, generating plots, and building dashboards.
- **src/region_selection/**: Utilities for selecting, defining, and managing geographic regions of interest.
- **src/data_retrieval/**: Handles data fetching from external sources such as NASA APIs and other environmental datasets.
- **src/preprocessing/**: Functions and scripts for cleaning, validating, and preparing data for analysis and modeling.
- **src/modeling/**: Machine learning models, training routines, and related utilities for predictive analytics.
- **src/insights/**: Modules for generating actionable insights and reports from processed and modeled data.

- **config/**: YAML configuration files for customizing agent behavior, RAG, MCP, and overall system settings.
- **templates/**: HTML templates for rendering the web interface.
- **static/**: Static files (CSS, JavaScript, images) used by the web interface.
- **tests/**: Automated tests to ensure code quality and correctness.
- **logs/**: Log files generated during application execution.
- **reports/**: Output reports and summaries generated by the system.
- **venv/**: Python virtual environment directory (should be excluded from version control).
- **requirements.txt**: Lists all Python dependencies required to run the project.
- **main.py**: The main script to launch the SEMRIS application.
- **README.md**: This documentation file.
- **project_structure.md**: A standalone, detailed reference of the project structure.

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
- Review and modify configuration files in the `config/` directory
- Set up API keys for external services
- Configure AI agent parameters as needed

## Usage

1. Run the application:
```bash
python main.py
```

The application will:
- Initialize the AI agent system
- Select the Bengaluru region (configurable)
- Retrieve environmental data
- Process and analyze data using AI agents
- Launch the web interface

2. Access the web interface:
- Open your browser and navigate to `http://localhost:5000`
- View environmental data and predictions
- Interact with AI agents through the interface
- Monitor agent performance and learning

## Dependencies

Key dependencies include:
- LangChain (Agent framework)
- PyTorch (Deep learning)
- Sentence Transformers (Embeddings)
- ChromaDB (Vector store)
- Flask (Web interface)
- NASA API (Data retrieval)
- Redis (Agent communication)
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
- LangChain for agent framework
- PyTorch for deep learning
- ChromaDB for vector storage
- All other open-source contributors 