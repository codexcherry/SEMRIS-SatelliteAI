"""
Memory Manager Module for Model Context Protocol (MCP).
Handles state persistence and memory management for AI agents.
"""

import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np

class MemoryManager:
    """
    Manages memory and state persistence for AI agents.
    Provides mechanisms for storing and retrieving agent states,
    historical data, and learned patterns.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.memory = {
            'short_term': {},
            'long_term': {},
            'patterns': {},
            'metadata': {
                'last_update': None,
                'version': '1.0.0'
            }
        }
    
    def store_state(
        self,
        state: Dict[str, Any],
        memory_type: str = 'short_term'
    ) -> None:
        """
        Store state information in memory.
        
        Args:
            state: State information to store
            memory_type: Type of memory to use ('short_term' or 'long_term')
        """
        if memory_type not in ['short_term', 'long_term']:
            raise ValueError("memory_type must be 'short_term' or 'long_term'")
        
        # Update memory
        timestamp = datetime.now().isoformat()
        self.memory[memory_type][timestamp] = state
        
        # Cleanup old memories if needed
        if memory_type == 'short_term':
            self._cleanup_short_term()
        
        # Update metadata
        self.memory['metadata']['last_update'] = timestamp
    
    def retrieve_state(
        self,
        timestamp: Optional[str] = None,
        memory_type: str = 'short_term'
    ) -> Dict[str, Any]:
        """
        Retrieve state information from memory.
        
        Args:
            timestamp: Specific timestamp to retrieve (None for latest)
            memory_type: Type of memory to use
            
        Returns:
            Retrieved state information
        """
        if memory_type not in ['short_term', 'long_term']:
            raise ValueError("memory_type must be 'short_term' or 'long_term'")
        
        if not self.memory[memory_type]:
            return {}
        
        if timestamp is None:
            # Get latest state
            latest_timestamp = max(self.memory[memory_type].keys())
            return self.memory[memory_type][latest_timestamp]
        
        return self.memory[memory_type].get(timestamp, {})
    
    def store_pattern(
        self,
        pattern_id: str,
        pattern_data: Dict[str, Any]
    ) -> None:
        """
        Store a learned pattern.
        
        Args:
            pattern_id: Unique identifier for the pattern
            pattern_data: Pattern information
        """
        self.memory['patterns'][pattern_id] = {
            **pattern_data,
            'timestamp': datetime.now().isoformat()
        }
    
    def retrieve_patterns(
        self,
        pattern_ids: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Retrieve stored patterns.
        
        Args:
            pattern_ids: List of pattern IDs to retrieve (None for all)
            
        Returns:
            Dictionary of pattern data
        """
        if pattern_ids is None:
            return self.memory['patterns']
        
        return {
            pid: self.memory['patterns'][pid]
            for pid in pattern_ids
            if pid in self.memory['patterns']
        }
    
    def save_to_disk(self, path: str) -> None:
        """
        Save memory state to disk.
        
        Args:
            path: Path to save the memory state
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_memory = self._make_serializable(self.memory)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save to file
        with open(path, 'w') as f:
            json.dump(serializable_memory, f, indent=2)
    
    def load_from_disk(self, path: str) -> None:
        """
        Load memory state from disk.
        
        Args:
            path: Path to load the memory state from
        """
        if not os.path.exists(path):
            return
        
        with open(path, 'r') as f:
            self.memory = json.load(f)
    
    def _cleanup_short_term(self):
        """
        Clean up old short-term memories based on configuration.
        """
        max_items = self.config.get('max_short_term_items', 1000)
        if len(self.memory['short_term']) > max_items:
            # Sort by timestamp and keep only the most recent
            sorted_items = sorted(
                self.memory['short_term'].items(),
                key=lambda x: x[0]
            )
            self.memory['short_term'] = dict(sorted_items[-max_items:])
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON serializable format.
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj 