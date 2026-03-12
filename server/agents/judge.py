"""
Judge Agent - Critiques metrics & selects the production-ready winner
Evaluates model performance and makes final recommendations
"""

import asyncio
from typing import Dict
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage

from tools.executor import execute_in_parallel
from graph.state import AgentState
from utils.config import settings


llm = ChatGroq(model=settings.llm_model, temperature=0.1)

async def judge_agent(state: AgentState) -> Dict:

    execution_results = await execute_in_parallel(state.candidate_scripts)

    successful_runs = [
        res for res in execution_results if res.get("status") == "success"
    ]

    if not successful_runs:
        return {
            "messages": [
                AIMessage(content="No candidate scripts executed successfully. Cannot determine a winner.")
            ],
            "next_step": "end"
        }

    # Pick the run with the highest accuracy
    winning_run = max(
        successful_runs,
        key=lambda x: x["execution"]["results"].get("accuracy", 0)
    )

    # Find the matching script object by approach_name
    winning_script_obj = next(
        (s for s in state.candidate_scripts if s["approach_name"] == winning_run["approach_name"]),
        None
    )

    champion_bundle = {
        "approach_name": winning_run["approach_name"],
        "accuracy": winning_run["execution"]["results"].get("accuracy"),
        "metrics": winning_run["execution"]["results"],
        "code": winning_script_obj["code"] if winning_script_obj else "Code not found"
    }

    prompt = f"""
    You are the Lead Judge at Agnostos Lab. 
    Review the champion: {champion_bundle['approach_name']} 
    With Metrics: {champion_bundle['metrics']}
    
    Provide a 2-sentence final verdict:
    1. Confirm why this specific approach and its metrics make it the best choice.
    2. Mention that the source code has been successfully archived for deployment.
    """
    verdict_summary = llm.invoke(prompt).content

    return {
        "final_output": champion_bundle,
        "messages": [AIMessage(content=f"Judge Verdict: {verdict_summary}")],
        "next_step": "completed"
    }