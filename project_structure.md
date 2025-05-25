# Enhanced SEMRIS Project Structure

```
SEMRIS-SatelliteAI/
├── src/
│   ├── ai_agent/                  # AI Agent Components
│   │   ├── __init__.py
│   │   ├── base_agent.py          # Base agent class
│   │   ├── environmental_agent.py  # Environmental data analysis agent
│   │   ├── prediction_agent.py     # Prediction and forecasting agent
│   │   └── coordination_agent.py   # Agent coordination and communication
│   │
│   ├── agents/                    # Legacy agent components
│   │   └── __init__.py
│   │
│   ├── rag/                       # Retrieval Augmented Generation
│   │   ├── __init__.py
│   │   ├── document_store.py      # Vector store for environmental data
│   │   ├── retriever.py          # Context retrieval system
│   │   └── generator.py          # Response generation system
│   │
│   ├── mcp/                       # Model Context Protocol
│   │   ├── __init__.py
│   │   ├── context_manager.py     # Session and context management
│   │   ├── memory.py             # Memory management system
│   │   └── state_tracker.py      # State tracking and persistence
│   │
│   ├── adk/                       # Agent Development Kit
│   │   ├── __init__.py
│   │   ├── learning.py           # Agent learning mechanisms
│   │   ├── behavior.py           # Behavior adjustment system
│   │   └── metrics.py            # Performance tracking
│   │
│   ├── visualization/            # Data visualization components
│   │   └── __init__.py
│   │
│   ├── region_selection/         # Region selection utilities
│   │   └── __init__.py
│   │
│   ├── data_retrieval/           # Data retrieval components
│   │   └── __init__.py
│   │
│   ├── preprocessing/            # Data preprocessing utilities
│   │   └── __init__.py
│   │
│   ├── modeling/                 # ML modeling components
│   │   └── __init__.py
│   │
│   └── insights/                 # Analysis and insights generation
│       └── __init__.py
│
├── config/                       # Configuration files
│   ├── agent_config.yaml         # AI agent configuration
│   ├── rag_config.yaml          # RAG system configuration
│   ├── mcp_config.yaml          # MCP configuration
│   └── config.yaml              # General configuration
│
├── templates/                    # Web interface templates
├── static/                      # Static assets
├── tests/                       # Test suite
├── logs/                        # System logs
├── reports/                     # Analysis reports
├── venv/                        # Virtual environment
├── requirements.txt             # Project dependencies
├── main.py                      # Application entry point
├── README.md                    # Project documentation
└── project_structure.md         # This file
``` 