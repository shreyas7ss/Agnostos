"""
Executor Tool - Local subprocess execution of generated ML code
Runs candidate scripts in parallel using Python subprocesses (dev mode).
For production, swap this with Modal remote execution.
"""

import json
import asyncio
import tempfile
import os
import sys


async def _run_single_script(code_string: str, timeout: int = 120) -> dict:
    """Run a single ML script in an isolated subprocess and return its metrics."""
    # Write code to a temp file so we can exec it cleanly
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        # Wrap the code to capture metrics.json from same dir as the temp script
        f.write(code_string)
        tmp_path = f.name

    metrics_path = tmp_path.replace(".py", "_metrics.json")

    # Inject the metrics output path so the script writes to a known location
    patched_code = code_string.replace(
        "metrics.json", metrics_path
    )
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(patched_code)

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
            return {"status": "error", "message": f"Execution timed out after {timeout}s"}

        if proc.returncode != 0:
            return {"status": "error", "message": stderr.decode()[:500]}

        # Try reading metrics.json (original or patched path)
        for path in [metrics_path, "metrics.json"]:
            if os.path.exists(path):
                with open(path, "r") as mf:
                    results = json.load(mf)
                os.remove(path)
                return {"status": "success", "results": results}

        return {"status": "partial_success", "message": "metrics.json not written"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def execute_in_parallel(scripts: list[dict], timeout: int = 120) -> list[dict]:
    """Execute candidate scripts in parallel via local subprocesses."""
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
