"""
Profiler Agent - Performs deep EDA on data
Handles Tabular statistics and Image VLM analysis
"""

import os
from typing import Dict
from langchain_groq import ChatGroq
from tools.stats import tabular_profiler 
from tools.vision import image_profiler
from graph.state import AgentState
from utils.config import settings
from pathlib import Path
from langchain_core.messages import AIMessage


llm= ChatGroq(model=settings.llm_model, temperature=0.1)

def profiler_agent(state: AgentState) -> AgentState:
    path= state.dataset_path
    print(f"🔎 Profiler investigating: {path}")

    if os.path.isdir(path) or any(path.lower().endswith(ext) for ext in ['.jpg', '.png', '.jpeg']):
        manifesto = image_profiler(path)
        data_type = "image"
    else:
        manifesto = tabular_profiler(path)
        data_type = "tabular"

    prompt=f"""
     You are a Data Science Consultant. Review this {data_type} manifesto:
     {manifesto}
    
     Provide a 2-sentence summary for the Scientist agent:
     1. Highlight the most critical data quality finding.
     2. Suggest a strategy for the upcoming parallel experiments.
     """
    consultant_summary=llm.invoke(prompt).content

    return{
     "data_manifesto":{**manifesto,"data_type":data_type},
    "messages":[AIMessage(content=f"PROFILER BREIFING {consultant_summary}")],
    "next_step":"scientist"
}
        
