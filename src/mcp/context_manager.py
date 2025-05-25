"""
Context Manager Module for Model Context Protocol (MCP).
Manages context and state for AI agents.
"""

import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
from .memory import MemoryManager

class ContextManager:
    """
    Manages context and state for AI agents.
    Provides mechanisms for state persistence, context tracking,
    and session management.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.memory_manager = MemoryManager(config)
        self.current_session = {
            'id': datetime.now().isoformat(),
            'start_time': datetime.now().isoformat(),
            'context': {},
            'state': {}
        }
    
    def update_context(
        self,
        context_data: Dict[str, Any],
        context_type: str = 'analysis'
    ) -> None:
        """
        Update session context with new data.
        
        Args:
            context_data: New context data to add
            context_type: Type of context being updated
        """
        if context_type not in self.current_session['context']:
            self.current_session['context'][context_type] = {}
        
        self.current_session['context'][context_type].update(
            context_data
        )
        
        # Store in memory
        self.memory_manager.store_state(
            {
                'session_id': self.current_session['id'],
                'context_type': context_type,
                'context_data': context_data,
                'timestamp': datetime.now().isoformat()
            },
            memory_type='short_term'
        )
    
    def get_context(
        self,
        context_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current session context.
        
        Args:
            context_type: Specific type of context to retrieve
            
        Returns:
            Current context data
        """
        if context_type:
            return self.current_session['context'].get(context_type, {})
        return self.current_session['context']
    
    def update_state(
        self,
        state_data: Dict[str, Any],
        state_type: str = 'analysis'
    ) -> None:
        """
        Update session state with new data.
        
        Args:
            state_data: New state data
            state_type: Type of state being updated
        """
        if state_type not in self.current_session['state']:
            self.current_session['state'][state_type] = {}
        
        self.current_session['state'][state_type].update(
            state_data
        )
        
        # Store in memory
        self.memory_manager.store_state(
            {
                'session_id': self.current_session['id'],
                'state_type': state_type,
                'state_data': state_data,
                'timestamp': datetime.now().isoformat()
            },
            memory_type='long_term'
        )
    
    def get_state(
        self,
        state_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current session state.
        
        Args:
            state_type: Specific type of state to retrieve
            
        Returns:
            Current state data
        """
        if state_type:
            return self.current_session['state'].get(state_type, {})
        return self.current_session['state']
    
    def save_state(self, state: Dict[str, Any], path: str) -> None:
        """
        Save state to disk.
        
        Args:
            state: State data to save
            path: Path to save the state
        """
        # Update session state
        self.update_state(state)
        
        # Save to disk using memory manager
        self.memory_manager.save_to_disk(path)
    
    def load_state(self, path: str) -> Dict[str, Any]:
        """
        Load state from disk.
        
        Args:
            path: Path to load the state from
            
        Returns:
            Loaded state data
        """
        # Load from disk using memory manager
        self.memory_manager.load_from_disk(path)
        
        # Get latest state
        latest_state = self.memory_manager.retrieve_state(
            memory_type='long_term'
        )
        
        if latest_state:
            # Update current session state
            state_data = latest_state.get('state_data', {})
            self.update_state(state_data)
            return state_data
        
        return {}
    
    def start_new_session(self) -> None:
        """
        Start a new session with fresh context.
        """
        self.current_session = {
            'id': datetime.now().isoformat(),
            'start_time': datetime.now().isoformat(),
            'context': {},
            'state': {}
        }
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about the current session.
        
        Returns:
            Session information
        """
        return {
            'session_id': self.current_session['id'],
            'start_time': self.current_session['start_time'],
            'context_types': list(self.current_session['context'].keys()),
            'state_types': list(self.current_session['state'].keys())
        } 