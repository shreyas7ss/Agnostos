"""
Executor Tool - Self-Healing Local Subprocess Executor
Features:
  1. Auto-installs missing packages (from pyproject.toml deps)
  2. LLM feedback loop to fix broken generated code (up to max_retries)
"""

import json
import asyncio
import tempfile
import os
import re
import subprocess
import sys
from langchain_groq import ChatGroq
from utils.config import settings


# ── LLM for code fixing ──────────────────────────────────────────────────────
_fixer_llm = ChatGroq(
    model=settings.llm_model,
    api_key=settings.groq_api_key,
    temperature=0.2,
)


def _extract_missing_module(stderr: str) -> str | None:
    """Parse a ModuleNotFoundError to get the package name."""
    match = re.search(r"No module named '([^']+)'", stderr)
    if match:
        # e.g. "sklearn" → "scikit-learn", "imblearn" → "imbalanced-learn"
        alias_map = {
            "sklearn": "scikit-learn",
            "imblearn": "imbalanced-learn",
            "cv2": "opencv-python",
            "PIL": "pillow",
            "bs4": "beautifulsoup4",
            "yaml": "pyyaml",
        }
        raw = match.group(1).split(".")[0]
        return alias_map.get(raw, raw)
    return None


def _install_package(pkg: str) -> bool:
    """Install a package into the current environment using uv add or pip."""
    print(f"[EXECUTOR] Auto-installing missing package: {pkg}")
    try:
        result = subprocess.run(
            ["uv", "add", pkg],
            capture_output=True, text=True, timeout=60,
            cwd=os.path.dirname(os.path.dirname(__file__))   # project root
        )
        if result.returncode == 0:
            print(f"[EXECUTOR] ✓ Installed {pkg}")
            return True
        # fallback to pip
        result2 = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            capture_output=True, text=True, timeout=60
        )
        return result2.returncode == 0
    except Exception as e:
        print(f"[EXECUTOR] Install failed: {e}")
        return False


async def _ask_llm_to_fix(code: str, error: str) -> str:
    """Ask the LLM to fix the broken script given the traceback."""
    prompt = f"""You are a Python debugging expert. 
Fix the following ML script that failed with this error. 
Return ONLY the corrected Python code, no explanation, no markdown fences.

ERROR:
{error[:1500]}

BROKEN CODE:
{code[:3000]}
"""
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _fixer_llm.invoke(prompt)
        )
        return response.content.strip().removeprefix("```python").removesuffix("```").strip()
    except Exception as e:
        print(f"[EXECUTOR] LLM fix failed: {e}")
        return code   # return original if LLM fails


async def _run_script(code_string: str, timeout: int) -> tuple[bool, str, str]:
    """Write code to temp file and run it. Returns (success, stdout, stderr)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code_string)
        tmp_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return False, "", f"Timed out after {timeout}s"

        return proc.returncode == 0, stdout.decode(), stderr.decode()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def _run_single_script(code_string: str, timeout: int = 120) -> dict:
    """
    Run a script with self-healing:
      Round 1: Run as-is
      Round 2: Auto-install missing package if ModuleNotFoundError, retry
      Round 3: Send error to LLM for a code fix, retry
    """
    metrics_path = tempfile.mktemp(suffix="_metrics.json").replace("\\", "/")
    max_attempts = settings.max_retries  # default 3
    current_code = code_string.replace("metrics.json", metrics_path)

    for attempt in range(1, max_attempts + 1):
        print(f"[EXECUTOR] Attempt {attempt}/{max_attempts}...")
        ok, stdout, stderr = await _run_script(current_code, timeout)

        if ok:
            for path in [metrics_path, "metrics.json"]:
                if os.path.exists(path):
                    with open(path, "r") as mf:
                        results = json.load(mf)
                    os.remove(path)
                    print(f"[EXECUTOR] ✓ Success on attempt {attempt}")
                    return {"status": "success", "results": results}
            # Script ran but wrote no metrics — send to LLM
            no_metrics_err = f"Script ran OK but did not write metrics.json.\nstdout:\n{stdout[:400]}"
            print(f"[EXECUTOR] {no_metrics_err}")
            if attempt < max_attempts:
                print(f"[EXECUTOR] Sending to LLM for fix (attempt {attempt})...")
                fixed = await _ask_llm_to_fix(current_code, no_metrics_err)
                current_code = fixed.replace("metrics.json", metrics_path)
            continue

        print(f"[EXECUTOR ERROR] Attempt {attempt} failed:\n{stderr[:600]}")

        if attempt < max_attempts:
            # Try auto-install first
            pkg = _extract_missing_module(stderr)
            if pkg:
                installed = _install_package(pkg)
                if installed:
                    # re-patch in case install path changed
                    current_code = code_string.replace("metrics.json", metrics_path)
                    continue

            # Otherwise send to LLM for fix
            print(f"[EXECUTOR] Sending to LLM for fix (attempt {attempt})...")
            fixed = await _ask_llm_to_fix(current_code, stderr)
            current_code = fixed.replace("metrics.json", metrics_path)

    return {"status": "error", "message": f"Failed after {max_attempts} attempts. Last error:\n{stderr[:400]}"}


async def execute_in_parallel(scripts: list[dict], timeout: int = 120) -> list[dict]:
    """Execute candidate scripts in parallel with self-healing."""
    tasks = [_run_single_script(s["code"], timeout=timeout) for s in scripts]
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
