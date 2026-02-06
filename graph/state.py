"""
State Definition - Shared memory State (TypedDict) for inter-agent context
Defines the state schema for LangGraph workflow
"""
from typing import TypedDict, List, Optional, Annotated,Dict
from pydantic import BaseModel,Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage],add_messages]=Field(default_factory=list)

    experiment_id: Optional[str] =None

    dataset_path: Optional[str] =None

    data_manifesto:Optional[dict]=Field(default_factory=dict)
    
    next_step:Optional[str] =None

    candidate_scripts: List[Dict] =Field(default_factory=list)

    final_output : Optional[Dict] = None
    

