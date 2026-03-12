"""
Executor Tool - Modal Cloud Execution with Self-Healing
Features:
  1. Runs generated ML scripts in Modal cloud containers (deployed, not ephemeral)
  2. LLM feedback loop to fix broken code (up to max_retries)
"""

import asyncio
import re
from langchain_groq import ChatGroq
import modal

# ── Modal container image ─────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim()
    .pip_install(
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "imbalanced-learn",
        "requests",       # for URL downloads inside container
        "langchain_groq", # required for executor imports
        "langchain_core"  # required for executor imports
    )
)

app = modal.App("agnostas-lab-runner")

@app.function(image=image, timeout=1200)
def run_ml_code(code_string: str) -> dict:
    """
    Executes a generated ML script inside a Modal cloud container.
    The script must write results to 'metrics.json'.
    Dataset is loaded inside the script directly from a public URL (pd.read_csv works).
    """
    import os, json

    namespace = {}
    try:
        exec(code_string, namespace)

        if os.path.exists("metrics.json"):
            with open("metrics.json") as f:
                return {"status": "success", "results": json.load(f)}
        return {"status": "error", "message": "metrics.json was not written by the script"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# ── Reference to the DEPLOYED Modal function ─────────────────────────────────
# Use from_name() so the server can call the deployed function without needing
# a running Modal App context (avoids the "not hydrated" error).
_run_ml_code = modal.Function.from_name("agnostas-lab-runner", "run_ml_code")


# ── LLM for code fixing ───────────────────────────────────────────────────────
def _get_fixer_llm():
    from utils.config import settings
    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0.2,
    )

async def _ask_llm_to_fix(code: str, error: str) -> str:
    """Ask the LLM to fix the broken script given the traceback."""
    prompt = f"""You are a Python debugging expert.
Fix the following ML script that failed with this error.
The script MUST:
- Load its dataset from a URL using pd.read_csv(url) or similar.
- Save results as a JSON file called 'metrics.json' containing at least {{"accuracy": value, "f1_score": value, "recall": value, "precision": value}}.
Return ONLY corrected Python code. No markdown fences, no explanation.

ERROR:
{error[:1500]}

BROKEN CODE:
{code[:3000]}
"""
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _get_fixer_llm().invoke(prompt)
        )
        fixed = response.content.strip()
        fixed = re.sub(r"^```python\s*", "", fixed)
        fixed = re.sub(r"\s*```$", "", fixed)
        return fixed.strip()
    except Exception as e:
        print(f"[EXECUTOR] LLM fix failed: {e}")
        return code


async def _run_single_script(code_string: str) -> dict:
    """Run a script via Modal with self-healing retry loop."""
    from utils.config import settings
    max_attempts = settings.max_retries
    current_code = code_string

    for attempt in range(1, max_attempts + 1):
        print(f"[EXECUTOR] Attempt {attempt}/{max_attempts} (Modal)...")
        try:
            res = await _run_ml_code.remote.aio(current_code)
        except Exception as e:
            res = {"status": "error", "message": str(e)}

        if isinstance(res, dict) and res.get("status") == "success":
            print(f"[EXECUTOR] ✓ Success on attempt {attempt}")
            return res

        err_msg = res.get("message", "Unknown error") if isinstance(res, dict) else str(res)
        print(f"[EXECUTOR ERROR] Attempt {attempt} failed: {err_msg[:400]}")

        if attempt < max_attempts:
            print(f"[EXECUTOR] Sending to LLM for fix...")
            current_code = await _ask_llm_to_fix(current_code, err_msg)

    return {"status": "error", "message": f"Failed after {max_attempts} attempts. Last: {err_msg[:300]}"}


async def execute_in_parallel(scripts: list[dict], timeout: int = 120) -> list[dict]:
    """Execute candidate scripts in parallel via Modal cloud containers."""
    tasks = [_run_single_script(s["code"]) for s in scripts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [
        {
            "approach_name": script["approach_name"],
            "status": res.get("status", "error") if isinstance(res, dict) else "error",
            "execution": {
                "results": res.get("results", {}) if isinstance(res, dict) else {}
            },
        }
        for script, res in zip(scripts, results)
    ]


    return [
        {
            "approach_name": script["approach_name"],
            "status": res.get("status", "error") if isinstance(res, dict) else "error",
            "execution": {
                "results": res.get("results", {}) if isinstance(res, dict) else {}
            },
        }
        for script, res in zip(scripts, results)
    ]
