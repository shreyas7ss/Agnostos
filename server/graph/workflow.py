"""
Workflow Definition - LangGraph StateGraph
Defines Nodes, Edges, and Conditional Logic for agent orchestration
"""

from networkx.generators import graph_atlas
from langgraph.graph import StateGraph,END
from graph.state import AgentState
from agents.profiler import profiler_agent
from agents.executor import executor_agent
from agents.judge import judge_agent


def create_workflow():
    """
    Creates a workflow for the profiler, executor, and judge agents.
    """
    graph = StateGraph(AgentState())

    #add nodes 

    graph.add_node("profiler", profiler_agent)
    graph.add_node("executor", executor_agent)
    graph.add_node("judge", judge_agent)
  
    #add edges
    graph.set_entry_point("profiler")   
    graph.add_edge("profiler", "scientist")
    graph.add_edge("scientist", "judge")
    graph.add_edge("judge", END)

    app = graph.compile()

    return app

agnostos_graph = create_workflow()

    
    

    