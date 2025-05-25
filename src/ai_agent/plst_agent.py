"""
PLST Agent Module - Integrates AI agent system with PLST analysis.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
from src.modeling.plst_analyzer import PLSTAnalyzer
from src.rag.document_store import DocumentStore
from src.rag.retriever import Retriever
from src.mcp.context_manager import ContextManager
from src.mcp.memory import MemoryManager
from src.adk.learning import LearningModule

class PLSTAgent:
    """
    AI Agent specialized in PLST analysis and environmental monitoring.
    Combines PLST analysis with RAG, MCP, and ADK capabilities.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the PLST Agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Get persist directory from config
        persist_directory = self.config.get('document_store_path', 'data/chroma')
        
        # Initialize components
        self.plst_analyzer = PLSTAnalyzer(self.config)
        self.document_store = DocumentStore(persist_directory=persist_directory)
        self.retriever = Retriever(document_store=self.document_store)
        self.memory_manager = MemoryManager(self.config)
        self.context_manager = ContextManager(self.config)
        self.learning_module = LearningModule(self.config)
        
        # Initialize state
        self.state = {
            'last_analysis': None,
            'analysis_count': 0,
            'patterns_learned': 0
        }
        
        # Model version
        self.model_version = '2.0.0'
        
        # Load document store
        self.document_store.load()
    
    def analyze(
        self,
        input_data: Dict[str, Any],
        temporal_filter: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze environmental data with AI assistance.
        
        Args:
            input_data: Input data dictionary
            temporal_filter: Optional time window filter
            
        Returns:
            Analysis report
        """
        # Get historical context
        historical_context = self.retriever.get_relevant_context(
            query_context=input_data.get('query_context', {}),
            top_k=5
        )
        
        # Extract relevant patterns from historical context
        patterns = []
        for doc in historical_context:
            if isinstance(doc['content'], dict):
                patterns.extend(doc['content'].get('patterns', []))
        
        # Prepare data for analysis
        processed_data = self._preprocess_data(input_data['data'])
        
        # Perform PLST analysis
        plst_results = self.plst_analyzer.analyze(
            {
                'data': processed_data,
                'location': input_data.get('location', {}),
                'timestamp': input_data.get('timestamp'),
                'patterns': patterns
            },
            temporal_filter=temporal_filter
        )
        
        # Generate insights
        insights = self._generate_insights(plst_results, patterns)
        
        # Update memory with new analysis
        self._update_memory(plst_results, insights)
        
        return {
            'analysis': plst_results,
            'insights': insights,
            'historical_context': [
                {
                    'content': doc['content'],
                    'relevance': doc['score']
                }
                for doc in historical_context
            ],
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_version': self.model_version,
                'confidence': self._calculate_confidence(plst_results)
            }
        }
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current agent state.
        
        Returns:
            Current state information
        """
        return {
            **self.state,
            'context': self.context_manager.get_context(),
            'patterns': self.learning_module.get_current_patterns(),
            'confidence': self.learning_module.get_overall_confidence()
        }
    
    def save_state(self, path: str) -> None:
        """
        Save agent state to disk.
        
        Args:
            path: Path to save state
        """
        state_data = self.get_state()
        self.context_manager.save_state(state_data, path)
        self.document_store.save()
    
    def load_state(self, path: str) -> None:
        """
        Load agent state from disk.
        
        Args:
            path: Path to load state from
        """
        state_data = self.context_manager.load_state(path)
        if state_data:
            self.state = state_data.get('state', self.state)
            self.learning_module.load_patterns(
                state_data.get('patterns', {})
            )
        self.document_store.load()
    
    def _preprocess_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Preprocess input data.
        
        Args:
            data: Dictionary of parameter arrays
            
        Returns:
            Preprocessed data dictionary
        """
        processed = {}
        for param_name, param_data in data.items():
            # Ensure array is 2D
            if param_data.ndim == 1:
                processed[param_name] = param_data.reshape(-1, 1)
            elif param_data.ndim > 2:
                n_samples = param_data.shape[0]
                processed[param_name] = param_data.reshape(n_samples, -1)
            else:
                processed[param_name] = param_data
        return processed
        
    def _generate_insights(
        self,
        analysis_results: Dict[str, Any],
        patterns: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate insights from analysis results.
        
        Args:
            analysis_results: Analysis results dictionary
            patterns: List of historical patterns
            
        Returns:
            Dictionary of insights
        """
        insights = {
            'trends': [],
            'anomalies': [],
            'recommendations': []
        }
        
        # Extract trends
        if 'temporal_patterns' in analysis_results:
            temporal = analysis_results['temporal_patterns']
            if temporal:
                for trend_type, trend_value in temporal['trends'].items():
                    insights['trends'].append({
                        'type': trend_type,
                        'value': trend_value,
                        'confidence': temporal.get('seasonality', 0.5)
                    })
        
        # Extract anomalies
        if 'anomaly_scores' in analysis_results:
            anomaly_scores = analysis_results['anomaly_scores']
            if isinstance(anomaly_scores, list):
                for i, score in enumerate(anomaly_scores):
                    if score < -0.5:  # Significant anomaly
                        insights['anomalies'].append({
                            'index': i,
                            'score': score,
                            'severity': 'high' if score < -0.8 else 'medium'
                        })
        
        # Generate recommendations
        risk_scores = analysis_results.get('risk_scores', [])
        if len(risk_scores) > 0:
            mean_risk = float(np.mean(risk_scores))
            if mean_risk > 0.7:
                insights['recommendations'].append({
                    'priority': 'high',
                    'action': 'Immediate intervention required',
                    'details': 'High risk of environmental degradation detected'
                })
            elif mean_risk > 0.5:
                insights['recommendations'].append({
                    'priority': 'medium',
                    'action': 'Enhanced monitoring recommended',
                    'details': 'Moderate risk of environmental changes observed'
                })
        
        return insights
        
    def _update_memory(
        self,
        analysis_results: Dict[str, Any],
        insights: Dict[str, Any]
    ) -> None:
        """
        Update agent memory with new analysis results.
        
        Args:
            analysis_results: Analysis results dictionary
            insights: Generated insights
        """
        # Update state
        self.state['last_analysis'] = datetime.now().isoformat()
        self.state['analysis_count'] += 1
        
        # Store in memory - skip if memory_manager doesn't support add_entry
        memory_entry = {
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis_results,
            'insights': insights
        }
        try:
            if hasattr(self.memory_manager, 'add_entry'):
                self.memory_manager.add_entry(memory_entry)
            elif hasattr(self.memory_manager, 'add'):
                self.memory_manager.add(memory_entry)
            else:
                print("Warning: Memory manager doesn't support adding entries")
        except Exception as e:
            print(f"Error updating memory: {e}")
            # Continue without failing
        
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """
        Calculate confidence score for analysis results.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence_scores = []
        
        # Check data quality
        if 'metadata' in results:
            data_confidence = results['metadata'].get('confidence', 0.5)
            confidence_scores.append(data_confidence)
        
        # Check temporal patterns
        if 'temporal_patterns' in results:
            temporal = results['temporal_patterns']
            if temporal and 'seasonality' in temporal:
                confidence_scores.append(abs(temporal['seasonality']))
        
        # Check risk scores
        if 'risk_scores' in results:
            risk_scores = results['risk_scores']
            if len(risk_scores) > 0:
                score_std = float(np.std(risk_scores))
                score_range = float(np.ptp(risk_scores))
                if score_range > 0:
                    confidence_scores.append(min(1.0, score_std / (0.1 * score_range)))
        
        # Calculate overall confidence
        if confidence_scores:
            return float(np.mean(confidence_scores))
        return 0.5 