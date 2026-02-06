"""
Scientist Agent - Generates custom training logic
Supports YOLO, Transformers, and Tree-based models
"""

from http.client import responses
import json
from typing import List,Dict
from pydantic import BaseModel,Field
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq

from graph.state import AgentState
from utils.config import settings


class MLScript(BaseModel):
  approach_name: str = Field(description="name of the model or stratergy")
  explanation: str = Field(description="breif reasoning for choosing this stratergy")
  code: str = Field(description="The COMPLETE, standalone Python code to load data, train, and save metrics.json.")


class ScienctistProposals(BaseModel):
    experiments : List[MLScript] = Field(description="A list of distinct parallel proposals. ")

#this agent goes through the manifesto and genrates multiple experiments and its complete scripts
def scientist_agent(State : AgentState)->Dict:
    llm=ChatGroq(model=settings.llm_model,temperature=0.7)
    structured_llm=llm.with_structured_output(ScienctistProposals)

    manifesto=State.data_manifesto
    data_type=manifesto.get("data_type","tabular")
    num_attempts=settings.max_parallel_attempts

    if data_type=="image":
        task_specifics = "Use PyTorch. Focus on architectures like ResNet, EfficientNet, or ViT. Include standard image augmentations."
    else:
        task_specifics = "Use Scikit-Learn, XGBoost, or LightGBM. Focus on feature engineering, scaling, and handling imbalances."

    system_prompt = f"""
    You are a Senior Machine Learning Scientist at Agnostos Lab.
    Your goal is to propose {num_attempts} high-quality, diverse experiments to solve the task.
    
    DATA MANIFESTO:
    {json.dumps(manifesto, indent=2)}
    
    DATASET PATH: {State.dataset_path}
    
    INSTRUCTIONS:
    1. Propose EXACTLY {num_attempts} different experiments.
    2. {task_specifics}
    3. Each 'code' block must be a complete script that:
       - Loads data from '{State.dataset_path}'.
       - Preprocesses data correctly based on the manifesto.
       - Trains the model.
       - Saves a 'metrics.json' file containing at least '{{"accuracy": value, "loss": value}}'.
    4. Ensure the scripts are standalone and require no external input.
    """

    try:
        responses = structured_llm.invoke(system_prompt)
        candidate_scripts = [exp.model_dump() for exp in responses.experiments] # converting the pydantic object to a class 
        candidate_scripts=candidate_scripts[:num_attempts]
    
        summary_text = f"Proposing {len(candidate_scripts)} strategies: " + \
                      ", ".join([s['approach_name'] for s in candidate_scripts])
  
    except:
        candidate_scripts=[]
        summary_text="Scientist failed to genrate valid proposal"

    return{                          #updating the the global state
        "candidate_scripts":candidate_scripts,
        "messages":[AIMessage(content=summary_text)],
        "next_step":"judge"
    }
