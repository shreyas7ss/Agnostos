"""
FastAPI routes for Agnostos Lab
"""

import os
from supabase import create_client
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.config import settings

from graph.workflow import agnostos_graph
from database.session import SessionLocal
from database.models import Experiment, AgentStep
from database.init_db import init_db


app = FastAPI(title="Agnostos Lab API", version="1.0.0")

# CORS — allows the browser client to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Create DB tables on first run."""
    init_db()


# Supabase client — use service_role key for server-side uploads (bypasses RLS)
_supabase_key = settings.supabase_service_key or settings.supabase_key
_supabase = create_client(settings.supabase_url, _supabase_key)



class ExperimentRequest(BaseModel):
    dataset_path: str
    target_column: str


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Uploads a dataset to Supabase Storage and returns its public URL."""
    file_bytes = await file.read()
    storage_path = f"uploads/{file.filename}"

    # Upload (upsert so re-uploads work)
    _supabase.storage.from_(settings.supabase_bucket).upload(
        path=storage_path,
        file=file_bytes,
        file_options={"upsert": "true", "content-type": file.content_type or "text/csv"},
    )

    public_url = _supabase.storage.from_(settings.supabase_bucket).get_public_url(storage_path)
    return {"dataset_path": public_url}


async def run_agnostas_task(experiment_id: int, dataset_path: str, target_column: str):
    """Background task: runs the LangGraph workflow and streams updates to DB."""
    db = SessionLocal()
    try:
        initial_state = {
            "experiment_id": str(experiment_id),
            "dataset_path": dataset_path,
            "data_manifesto": {"target_column": target_column},
            "messages": []
        }

        async for output in agnostos_graph.astream(initial_state):
            for node_name, state_update in output.items():
                # Strip non-JSON-serializable values (e.g. AIMessage objects in 'messages')
                safe_details = {
                    k: v for k, v in state_update.items()
                    if k != "messages" and isinstance(v, (str, int, float, bool, dict, list, type(None)))
                }
                new_step = AgentStep(
                    experiment_id=experiment_id,
                    agent_name=node_name,
                    thought=str(state_update.get("messages", ["Node finished"])[-1].content),
                    details=safe_details
                )
                db.add(new_step)
                db.commit()

        # Mark complete only after the whole graph finishes
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if experiment:
            experiment.status = "COMPLETED"
            db.commit()

    except Exception as e:
        print(f"Error in background task: {e}")
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if experiment:
            experiment.status = "FAILED"
            db.commit()
    finally:
        db.close()


# ── Routes ──────────────────────────────────────────────────

@app.post("/experiment/start")
async def start_experiment(req: ExperimentRequest, background_tasks: BackgroundTasks):
    """Starts a new parallel ML experiment."""
    db = SessionLocal()
    try:
        new_exp = Experiment(status="RUNNING", dataset_path=req.dataset_path)
        db.add(new_exp)
        db.commit()
        db.refresh(new_exp)

        background_tasks.add_task(
            run_agnostas_task, new_exp.id, req.dataset_path, req.target_column
        )

        return {"experiment_id": new_exp.id, "status": "Started"}
    finally:
        db.close()


@app.get("/experiments/{exp_id}")
async def get_status(exp_id: int):
    """Fetches the current status and agent logs of an experiment."""
    db = SessionLocal()
    try:
        exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")

        steps = db.query(AgentStep).filter(AgentStep.experiment_id == exp_id).all()

        return {
            "status": exp.status,
            "logs": [{"agent": s.agent_name, "message": s.thought} for s in steps],
            "final_result": steps[-1].details.get("final_output") if exp.status == "COMPLETED" and steps else None
        }
    finally:
        db.close()