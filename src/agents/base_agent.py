"""
Base Agent implementation for the SEMRIS system.
This serves as the foundation for all specialized agents in the system.
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import asyncio
from pydantic import BaseModel
from langchain.llms import BaseLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class AgentState(BaseModel):
    """Represents the current state of an agent"""
    agent_id: str
    status: str
    context: Dict[str, Any]
    memory: Dict[str, Any]
    last_action: Optional[str] = None
    performance_metrics: Dict[str, float] = {}

class BaseAgent(ABC):
    """Base agent class implementing core agent functionality"""
    
    def __init__(
        self,
        agent_id: str,
        llm: BaseLLM,
        prompt_template: str,
        memory_key: str = "chat_history"
    ):
        self.agent_id = agent_id
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key=memory_key)
        self.state = AgentState(
            agent_id=agent_id,
            status="initialized",
            context={},
            memory={},
        )
        
        self.chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(prompt_template),
            memory=self.memory,
            verbose=True
        )
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results"""
        pass
    
    @abstractmethod
    async def learn(self, feedback: Dict[str, Any]) -> None:
        """Update agent behavior based on feedback"""
        pass
    
    async def update_state(self, new_state: Dict[str, Any]) -> None:
        """Update agent's internal state"""
        self.state.context.update(new_state.get("context", {}))
        self.state.memory.update(new_state.get("memory", {}))
        self.state.last_action = new_state.get("last_action", self.state.last_action)
        self.state.status = new_state.get("status", self.state.status)
    
    async def get_state(self) -> AgentState:
        """Return current agent state"""
        return self.state
    
    async def save_context(self, context: Dict[str, Any]) -> None:
        """Save context for future reference"""
        self.state.context.update(context)
    
    async def get_context(self) -> Dict[str, Any]:
        """Retrieve current context"""
        return self.state.context
    
    async def record_metric(self, metric_name: str, value: float) -> None:
        """Record a performance metric"""
        self.state.performance_metrics[metric_name] = value
    
    async def get_metrics(self) -> Dict[str, float]:
        """Get all recorded metrics"""
        return self.state.performance_metrics
    
    async def reset(self) -> None:
        """Reset agent state"""
        self.state = AgentState(
            agent_id=self.agent_id,
            status="reset",
            context={},
            memory={},
        )
        self.memory.clear()
    
    @abstractmethod
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data before processing"""
        pass
    
    @abstractmethod
    async def format_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format output data before returning"""
        pass 