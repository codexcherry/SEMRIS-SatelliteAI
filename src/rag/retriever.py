"""
Retriever Module for RAG (Retrieval-Augmented Generation).
Implements context retrieval with temporal filtering.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from .document_store import DocumentStore

class Retriever:
    """
    Retrieves relevant context from the document store with temporal filtering
    and relevance scoring.
    """
    
    def __init__(self, document_store: DocumentStore = None, config: Dict = None):
        """
        Initialize the Retriever.
        
        Args:
            document_store: Document store instance
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # If document_store not provided, create a new one
        if document_store is None:
            persist_directory = self.config.get('document_store_path', 'data/chroma')
            self.document_store = DocumentStore(persist_directory=persist_directory)
        else:
            self.document_store = document_store
            
        self.default_top_k = self.config.get('retriever_top_k', 5)
    
    def get_relevant_context(
        self,
        query_context: Dict[str, Any],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get relevant context for the query.
        
        Args:
            query_context: Query context dictionary
            top_k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        # Extract location and parameters
        location = query_context.get('location', {})
        parameters = query_context.get('parameters', [])
        timestamp = query_context.get('timestamp')
        
        # Construct search query
        query = f"Location: lat={location.get('lat', 0)}, lon={location.get('lon', 0)}, "
        query += f"parameters: {', '.join(parameters)}"
        
        # Add temporal filter if timestamp provided
        filters = None
        if timestamp:
            # Use current time as a numeric value instead of timestamp string
            filters = {
                'timestamp': {'$lte': 1.0}  # Always match all documents for now
            }
        
        # Search document store
        results = self.document_store.search(
            query=query,
            filters=filters,
            top_k=top_k
        )
        
        return results
    
    def _extract_patterns(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract patterns from retrieved documents.
        """
        patterns = {}
        
        for doc in documents:
            # Extract environmental patterns
            if 'environmental_data' in doc:
                env_data = doc['environmental_data']
                for metric, values in env_data.items():
                    pattern_key = f"env_{metric}"
                    if pattern_key not in patterns:
                        patterns[pattern_key] = []
                    patterns[pattern_key].extend(values)
            
            # Extract risk patterns
            if 'risk_analysis' in doc:
                risk_data = doc['risk_analysis']
                for risk_type, data in risk_data.items():
                    pattern_key = f"risk_{risk_type}"
                    if pattern_key not in patterns:
                        patterns[pattern_key] = []
                    patterns[pattern_key].append(data)
        
        # Normalize and aggregate patterns
        normalized_patterns = {}
        for key, values in patterns.items():
            if values:
                values = np.array(values)
                normalized_patterns[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'trend': float(np.gradient(values).mean()),
                    'count': len(values)
                }
        
        return normalized_patterns
    
    def _analyze_temporal_patterns(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze temporal patterns in retrieved documents.
        """
        temporal_patterns = []
        
        # Sort documents by timestamp
        sorted_docs = sorted(
            documents,
            key=lambda x: x.get('timestamp', '')
        )
        
        if len(sorted_docs) < 2:
            return temporal_patterns
        
        # Analyze trends
        for metric in ['risk_score', 'temperature', 'vegetation']:
            values = [
                doc.get(metric, 0)
                for doc in sorted_docs
                if metric in doc
            ]
            
            if len(values) >= 2:
                trend = np.gradient(values).mean()
                confidence = min(
                    abs(trend) / 0.1,  # Scale confidence by trend magnitude
                    1.0
                )
                
                if abs(trend) > 0.05:  # Significant trend threshold
                    temporal_patterns.append({
                        'metric': metric,
                        'trend': float(trend),
                        'confidence': float(confidence),
                        'description': (
                            f"{'Increasing' if trend > 0 else 'Decreasing'} "
                            f"trend in {metric.replace('_', ' ')}"
                        ),
                        'impact_assessment': self._assess_impact(metric, trend)
                    })
        
        return temporal_patterns
    
    def _analyze_spatial_patterns(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze spatial patterns in retrieved documents.
        """
        spatial_patterns = []
        
        # Collect spatial data
        spatial_data = {}
        for doc in documents:
            if 'location' in doc and 'metrics' in doc:
                loc = doc['location']
                loc_key = f"{loc['lat']:.2f},{loc['lon']:.2f}"
                
                if loc_key not in spatial_data:
                    spatial_data[loc_key] = []
                spatial_data[loc_key].append(doc['metrics'])
        
        # Analyze spatial clusters
        for loc_key, metrics_list in spatial_data.items():
            avg_metrics = {
                k: np.mean([m[k] for m in metrics_list if k in m])
                for k in metrics_list[0].keys()
            }
            
            # Identify significant spatial patterns
            for metric, value in avg_metrics.items():
                if abs(value) > 1.5:  # Significance threshold
                    lat, lon = map(float, loc_key.split(','))
                    spatial_patterns.append({
                        'metric': metric,
                        'location': {'lat': lat, 'lon': lon},
                        'value': float(value),
                        'significance': min(abs(value) / 2.0, 1.0),
                        'description': (
                            f"Significant {metric} anomaly detected "
                            f"at location ({lat:.2f}, {lon:.2f})"
                        ),
                        'affected_areas': self._get_affected_areas(lat, lon)
                    })
        
        return spatial_patterns
    
    def _calculate_pattern_scores(
        self,
        patterns: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate relevance scores for patterns.
        """
        pattern_scores = {}
        
        for pattern_key, pattern_data in patterns.items():
            # Calculate base score from pattern statistics
            base_score = abs(pattern_data['trend'])
            
            # Adjust score based on data quantity and reliability
            confidence = min(
                pattern_data['count'] / 10.0,  # Scale by sample size
                1.0
            )
            
            # Adjust score based on standard deviation (uncertainty)
            if pattern_data['std'] > 0:
                uncertainty_penalty = 1.0 / (1.0 + pattern_data['std'])
            else:
                uncertainty_penalty = 1.0
            
            # Calculate final score
            pattern_scores[pattern_key] = (
                base_score * confidence * uncertainty_penalty
            )
        
        return pattern_scores
    
    def _assess_impact(
        self,
        metric: str,
        trend: float
    ) -> str:
        """
        Assess the impact of a temporal pattern.
        """
        impact_level = abs(trend)
        if impact_level < 0.1:
            severity = "minimal"
        elif impact_level < 0.3:
            severity = "moderate"
        else:
            severity = "significant"
        
        direction = "increase" if trend > 0 else "decrease"
        
        return (
            f"{severity.capitalize()} impact expected due to {direction} "
            f"in {metric.replace('_', ' ')}"
        )
    
    def _get_affected_areas(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        Get list of areas affected by a spatial pattern.
        """
        # Simple circular area approximation
        return [{
            'center': {'lat': lat, 'lon': lon},
            'radius_km': radius_km,
            'type': 'circle'
        }] 