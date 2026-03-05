"""
Executor Tool - Interface for E2B/Modal sandboxed code execution
Provides secure execution environment for generated code
"""

import modal
import json
import asyncio
from utils.config import settings

image = (
    modal.Image.debian_slim()
    .pip_install(
        "pandas",
        "scikit-learn",
        "torch",
        "torchvision",
        "xgboost",
        "numpy"
    )
)

app = modal.App("agnostas-lab-runner")

@app.function(image=image, timeout=settings.execution_timeout)
def run_ml_code(code_string: str):
    """
    Executes raw python code in a remote Modal container.
    The code is expected to save results to 'metrics.json'.
    """
    import os
    import json

    exec_namespace = {}

    try:
        exec(code_string, exec_namespace)

        if os.path.exists("metrics.json"):
            with open("metrics.json", "r") as f:
                return {"status": "success", "results": json.load(f)}
        else:
            return {"status": "partial success", "message": "metrics.json not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def execute_in_parallel(scripts: list[dict]):
    """Execute candidate scripts in parallel via Modal and return results."""
    tasks = []

    for script in scripts:
        call = run_ml_code.remote.aio(script["code"])
        tasks.append(call)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [
        {
            "approach_name": script["approach_name"],
            "status": res.get("status", "error") if isinstance(res, dict) else "error",
            "execution": {"results": res.get("results", {})} if isinstance(res, dict) else {"results": {}},
        }
        for script, res in zip(scripts, results)
    ]
