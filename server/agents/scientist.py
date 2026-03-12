"""
Scientist Agent - Generates custom training logic
Supports YOLO, Transformers, and Tree-based models
"""

import json
import re
from typing import List, Dict
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq

from graph.state import AgentState
from utils.config import settings


def scientist_agent(State: AgentState) -> Dict:
    llm = ChatGroq(model=settings.llm_model, temperature=0.7)

    manifesto = State.data_manifesto
    data_type = manifesto.get("data_type", "tabular")
    num_attempts = settings.max_parallel_attempts

    if data_type == "image":
        task_specifics = "Use PyTorch. Focus on architectures like ResNet, EfficientNet, or ViT. Include standard image augmentations."
    else:
        task_specifics = "Use Scikit-Learn, XGBoost, or LightGBM. Focus on feature engineering, scaling, and handling imbalances."

    system_prompt = f"""You are a Senior Machine Learning Scientist at Agnostos Lab.
Propose EXACTLY {num_attempts} diverse ML experiments for this task.

DATA MANIFESTO:
{json.dumps(manifesto, indent=2)}

DATASET URL: {State.dataset_path}

INSTRUCTIONS:
1. {task_specifics}
2. Each script MUST:
   - Load data with: df = pd.read_csv('{State.dataset_path}')
   - Train a model on the target column in the manifesto.
   - Save results: json.dump({{"accuracy": float, "f1_score": float, "recall": float, "precision": float}}, open('metrics.json','w'))
3. Scripts must be fully standalone — no user input, no external files.

Return ONLY a valid JSON object in this exact format (no markdown, no explanation):
{{
  "experiments": [
    {{
      "approach_name": "Short strategy name",
      "explanation": "One sentence reasoning",
      "code": "import pandas as pd\\n..."
    }}
  ]
}}"""

    try:
        response = llm.invoke([{"role": "user", "content": system_prompt}])
        raw = response.content.strip()

        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        candidate_scripts = parsed.get("experiments", [])[:num_attempts]

        summary_text = f"Proposing {len(candidate_scripts)} strategies: " + \
                       ", ".join([s["approach_name"] for s in candidate_scripts])

    except Exception as e:
        print(f"[SCIENTIST ERROR] {e}")
        candidate_scripts = []
        summary_text = f"Scientist failed to generate valid proposal: {str(e)[:200]}"

    return {
        "candidate_scripts": candidate_scripts,
        "messages": [AIMessage(content=summary_text)],
        "next_step": "judge"
    }
