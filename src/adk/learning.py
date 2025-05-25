"""
Learning Module for Agent Development Kit (ADK).
Implements learning capabilities for AI agents.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

class LearningModule:
    """
    Implements learning capabilities for AI agents.
    Manages pattern recognition, recommendation generation,
    and continuous learning from interactions.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.patterns = {}
        self.recommendations = {}
        self.confidence_scores = {}
        self.learning_history = []
    
    def update_patterns(self, analysis_results: Dict[str, Any]) -> None:
        """
        Update learned patterns based on analysis results.
        
        Args:
            analysis_results: Results from analysis
        """
        timestamp = datetime.now().isoformat()
        
        # Extract patterns from analysis
        if 'patterns' in analysis_results:
            for pattern_key, pattern_data in analysis_results['patterns'].items():
                if pattern_key not in self.patterns:
                    self.patterns[pattern_key] = []
                
                self.patterns[pattern_key].append({
                    'timestamp': timestamp,
                    'data': pattern_data,
                    'confidence': pattern_data.get('confidence', 0.5)
                })
        
        # Extract insights
        if 'historical_insights' in analysis_results:
            for insight in analysis_results['historical_insights']:
                pattern_key = f"insight_{insight['type']}"
                if pattern_key not in self.patterns:
                    self.patterns[pattern_key] = []
                
                self.patterns[pattern_key].append({
                    'timestamp': timestamp,
                    'data': insight,
                    'confidence': insight.get('confidence', 0.5)
                })
        
        # Update confidence scores
        self._update_confidence_scores()
        
        # Record learning event
        self.learning_history.append({
            'timestamp': timestamp,
            'type': 'pattern_update',
            'patterns_updated': list(analysis_results.get('patterns', {}).keys()),
            'confidence_scores': self.confidence_scores.copy()
        })
    
    def generate_recommendations(
        self,
        current_scores: np.ndarray,
        component_indices: Dict[str, np.ndarray],
        historical_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on learned patterns.
        
        Args:
            current_scores: Current risk scores
            component_indices: Component analysis indices
            historical_context: Historical context data
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Analyze current situation
        situation = self._analyze_situation(
            current_scores,
            component_indices
        )
        
        # Generate recommendations based on patterns
        for pattern_key, pattern_instances in self.patterns.items():
            if not pattern_instances:
                continue
            
            # Get most recent pattern instance
            latest_pattern = pattern_instances[-1]['data']
            
            # Check if pattern is relevant to current situation
            if self._is_pattern_relevant(latest_pattern, situation):
                confidence = self._calculate_recommendation_confidence(
                    pattern_key,
                    situation
                )
                
                if confidence > 0.6:  # Confidence threshold
                    recommendation = self._create_recommendation(
                        pattern_key,
                        latest_pattern,
                        situation,
                        confidence
                    )
                    recommendations.append(recommendation)
        
        # Sort by confidence
        recommendations.sort(
            key=lambda x: x['confidence'],
            reverse=True
        )
        
        return recommendations
    
    def get_current_patterns(self) -> Dict[str, Any]:
        """
        Get current learned patterns.
        
        Returns:
            Dictionary of current patterns
        """
        current_patterns = {}
        
        for pattern_key, pattern_instances in self.patterns.items():
            if pattern_instances:
                current_patterns[pattern_key] = {
                    'latest': pattern_instances[-1]['data'],
                    'confidence': self.confidence_scores.get(pattern_key, 0.0),
                    'instance_count': len(pattern_instances)
                }
        
        return current_patterns
    
    def get_recommendation_confidence(self) -> float:
        """
        Get overall confidence in recommendations.
        
        Returns:
            Confidence score between 0 and 1
        """
        if not self.confidence_scores:
            return 0.0
        
        return float(np.mean(list(self.confidence_scores.values())))
    
    def get_overall_confidence(self) -> float:
        """
        Get overall confidence in learning system.
        
        Returns:
            Confidence score between 0 and 1
        """
        if not self.patterns:
            return 0.0
        
        # Consider multiple factors
        pattern_confidence = self.get_recommendation_confidence()
        pattern_diversity = len(self.patterns) / 10.0  # Normalize by expected number
        learning_progress = len(self.learning_history) / 100.0  # Normalize by expected history
        
        # Combine factors
        overall_confidence = (
            pattern_confidence * 0.5 +
            pattern_diversity * 0.3 +
            learning_progress * 0.2
        )
        
        return min(overall_confidence, 1.0)
    
    def load_patterns(self, patterns: Dict[str, Any]) -> None:
        """
        Load patterns from saved state.
        
        Args:
            patterns: Dictionary of patterns to load
        """
        self.patterns = {}
        timestamp = datetime.now().isoformat()
        
        for pattern_key, pattern_data in patterns.items():
            self.patterns[pattern_key] = [{
                'timestamp': timestamp,
                'data': pattern_data['latest'],
                'confidence': pattern_data.get('confidence', 0.5)
            }]
        
        self._update_confidence_scores()
    
    def _update_confidence_scores(self) -> None:
        """
        Update confidence scores for all patterns.
        """
        for pattern_key, pattern_instances in self.patterns.items():
            if not pattern_instances:
                continue
            
            # Calculate confidence based on:
            # 1. Number of instances
            instance_confidence = min(len(pattern_instances) / 10.0, 1.0)
            
            # 2. Consistency of observations
            consistency = self._calculate_pattern_consistency(pattern_instances)
            
            # 3. Recency of observations
            recency = self._calculate_pattern_recency(pattern_instances)
            
            # Combine factors
            self.confidence_scores[pattern_key] = (
                instance_confidence * 0.4 +
                consistency * 0.4 +
                recency * 0.2
            )
    
    def _calculate_pattern_consistency(
        self,
        pattern_instances: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate consistency score for pattern instances.
        """
        if len(pattern_instances) < 2:
            return 0.5
        
        # Extract confidence values
        confidences = [
            instance['confidence']
            for instance in pattern_instances
        ]
        
        # Calculate consistency score
        consistency = 1.0 - np.std(confidences)
        return float(max(0.0, consistency))
    
    def _calculate_pattern_recency(
        self,
        pattern_instances: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate recency score for pattern instances.
        """
        if not pattern_instances:
            return 0.0
        
        # Get timestamps
        timestamps = [
            datetime.fromisoformat(instance['timestamp'])
            for instance in pattern_instances
        ]
        
        # Calculate days since most recent
        days_since = (datetime.now() - max(timestamps)).days
        
        # Score decreases with age
        return float(max(0.0, 1.0 - (days_since / 30.0)))
    
    def _analyze_situation(
        self,
        current_scores: np.ndarray,
        component_indices: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Analyze current situation for recommendation generation.
        """
        return {
            'risk_level': float(np.mean(current_scores)),
            'risk_std': float(np.std(current_scores)),
            'component_scores': {
                name: float(np.mean(indices))
                for name, indices in component_indices.items()
            },
            'extreme_values': {
                name: float(np.sum(indices > 2.0))
                for name, indices in component_indices.items()
            }
        }
    
    def _is_pattern_relevant(
        self,
        pattern: Dict[str, Any],
        situation: Dict[str, Any]
    ) -> bool:
        """
        Check if pattern is relevant to current situation.
        """
        # Pattern is relevant if:
        # 1. Risk levels are similar
        if 'risk_level' in pattern:
            pattern_risk = pattern['risk_level']
            current_risk = situation['risk_level']
            if abs(pattern_risk - current_risk) > 0.3:
                return False
        
        # 2. Component scores show similar trends
        if 'component_scores' in pattern:
            pattern_scores = pattern['component_scores']
            current_scores = situation['component_scores']
            
            for component, score in pattern_scores.items():
                if component in current_scores:
                    if abs(score - current_scores[component]) > 0.3:
                        return False
        
        return True
    
    def _calculate_recommendation_confidence(
        self,
        pattern_key: str,
        situation: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence score for a recommendation.
        """
        # Base confidence from pattern
        base_confidence = self.confidence_scores.get(pattern_key, 0.0)
        
        # Adjust based on situation similarity
        pattern_instances = self.patterns[pattern_key]
        if pattern_instances:
            latest_pattern = pattern_instances[-1]['data']
            similarity = self._calculate_situation_similarity(
                latest_pattern,
                situation
            )
            
            # Combine scores
            confidence = (base_confidence + similarity) / 2.0
            return float(confidence)
        
        return 0.0
    
    def _calculate_situation_similarity(
        self,
        pattern: Dict[str, Any],
        situation: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between pattern and current situation.
        """
        similarities = []
        
        # Compare risk levels
        if 'risk_level' in pattern and 'risk_level' in situation:
            risk_sim = 1.0 - abs(
                pattern['risk_level'] - situation['risk_level']
            )
            similarities.append(risk_sim)
        
        # Compare component scores
        if 'component_scores' in pattern and 'component_scores' in situation:
            pattern_scores = pattern['component_scores']
            current_scores = situation['component_scores']
            
            for component in set(pattern_scores) & set(current_scores):
                score_sim = 1.0 - abs(
                    pattern_scores[component] - current_scores[component]
                )
                similarities.append(score_sim)
        
        if not similarities:
            return 0.0
        
        return float(np.mean(similarities))
    
    def _create_recommendation(
        self,
        pattern_key: str,
        pattern: Dict[str, Any],
        situation: Dict[str, Any],
        confidence: float
    ) -> Dict[str, Any]:
        """
        Create a recommendation based on pattern and situation.
        """
        # Determine severity
        severity = self._determine_severity(situation)
        
        # Generate recommendation details
        if pattern_key.startswith('env_'):
            category = 'Environmental Management'
            action = self._generate_environmental_action(pattern, situation)
        elif pattern_key.startswith('risk_'):
            category = 'Risk Mitigation'
            action = self._generate_risk_action(pattern, situation)
        else:
            category = 'General'
            action = self._generate_general_action(pattern, situation)
        
        return {
            'category': category,
            'priority': severity,
            'action': action,
            'confidence': float(confidence),
            'pattern_key': pattern_key,
            'details': (
                f"Based on {pattern_key} pattern with "
                f"{confidence:.1%} confidence"
            )
        }
    
    def _determine_severity(self, situation: Dict[str, Any]) -> str:
        """
        Determine recommendation severity based on situation.
        """
        risk_level = situation['risk_level']
        
        if risk_level > 0.8:
            return 'Critical'
        elif risk_level > 0.6:
            return 'High'
        elif risk_level > 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_environmental_action(
        self,
        pattern: Dict[str, Any],
        situation: Dict[str, Any]
    ) -> str:
        """
        Generate environmental management action.
        """
        component_scores = situation['component_scores']
        highest_impact = max(
            component_scores.items(),
            key=lambda x: x[1]
        )
        
        return (
            f"Implement environmental monitoring for {highest_impact[0]} "
            f"with focus on areas showing {highest_impact[1]:.1%} impact"
        )
    
    def _generate_risk_action(
        self,
        pattern: Dict[str, Any],
        situation: Dict[str, Any]
    ) -> str:
        """
        Generate risk mitigation action.
        """
        risk_level = situation['risk_level']
        extreme_values = situation['extreme_values']
        
        most_extreme = max(
            extreme_values.items(),
            key=lambda x: x[1]
        )
        
        return (
            f"Implement risk mitigation measures for {most_extreme[0]} "
            f"with {risk_level:.1%} risk level"
        )
    
    def _generate_general_action(
        self,
        pattern: Dict[str, Any],
        situation: Dict[str, Any]
    ) -> str:
        """
        Generate general action recommendation.
        """
        return (
            f"Monitor situation with {situation['risk_level']:.1%} "
            f"risk level and implement preventive measures"
        ) 