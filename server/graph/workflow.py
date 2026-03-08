"""
Workflow Definition - LangGraph StateGraph
Defines Nodes, Edges, and Conditional Logic for agent orchestration
"""

from langgraph.graph import StateGraph, END
from graph.state import AgentState
from agents.profiler import profiler_agent
from agents.scientist import scientist_agent
from agents.judge import judge_agent


def create_workflow():
    """
    Creates and compiles the Agnostos LangGraph workflow.
    Flow: profiler → scientist → judge → END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("profiler",  profiler_agent)
    graph.add_node("scientist", scientist_agent)
    graph.add_node("judge",     judge_agent)

    # Wire edges
    graph.set_entry_point("profiler")
    graph.add_edge("profiler",  "scientist")
    graph.add_edge("scientist", "judge")
    graph.add_edge("judge",     END)

    return graph.compile()


agnostos_graph = create_workflow()