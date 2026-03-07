import asyncio
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from graph.workflow import agnostos_graph
from database.session import SessionLocal
from database.models import Experiment, AgentStep
from utils.config import settings

app = FastAPI(title="Agnostos Lab API", version="1.0.0")

# --- 1. CORS Middleware (Essential for React) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your React URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Schemas ---
class ExperimentRequest(BaseModel):
    dataset_path: str
    target_column: str

# --- 3. The Corrected Background Task ---
async def run_agnostos_task(experiment_id: int, dataset_path: str, target_column: str):
    """
    Background worker that runs the LangGraph and persists every state 
    update to the Neon database as an AgentStep.
    """
    db = SessionLocal()
    try:
        initial_state = {
            "experiment_id": experiment_id,
            "dataset_path": dataset_path,
            "data_manifesto": {"target_column": target_column},
            "messages": []
        }

        # Use .astream to capture the output of each node (Profiler -> Scientist -> Judge)
        async for output in agnostos_graph.astream(initial_state):
            for node_name, state_update in output.items():
                
                # Extract the last message content for the "thought" column
                messages = state_update.get("messages", [])
                thought_text = messages[-1].content if messages else f"{node_name} completed task."

                # Create the step in DB
                new_step = AgentStep(
                    experiment_id=experiment_id,
                    agent_name=node_name,
                    thought=thought_text,
                    details=state_update # Fixed typo: 'detailes' -> 'details'
                )
                db.add(new_step)
                db.commit()

        # Update Experiment status to COMPLETED once the graph hits END
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if experiment:
            experiment.status = "COMPLETED"
            db.commit()

    except Exception as e:
        print(f"❌ Error in Agnostos Task: {e}")
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if experiment:
            experiment.status = "FAILED"
            db.commit()
    finally:
        db.close()

# --- 4. Routes ---

@app.post("/experiment/start")    
async def start_experiment(req: ExperimentRequest, background_tasks: BackgroundTasks):
    """Starts a new Parallel ML Experiment."""
    db = SessionLocal()
    try:
        # Create initial entry in Neon
        new_exp = Experiment(status="RUNNING", dataset_path=req.dataset_path)
        db.add(new_exp)
        db.commit()
        db.refresh(new_exp)
        
        # Fire and Forget the LangGraph
        background_tasks.add_task(
            run_agnostos_task, 
            new_exp.id, 
            req.dataset_path, 
            req.target_column
        )
        
        return {"experiment_id": new_exp.id, "status": "Started"}
    finally:
        db.close()

@app.get("/experiments/{exp_id}")
async def get_status(exp_id: int):
    """Fetches the current status and logs of an experiment for the React UI."""
    db = SessionLocal()
    try:
        exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        # Fetch all steps associated with this experiment
        steps = db.query(AgentStep).filter(AgentStep.experiment_id == exp_id).order_by(AgentStep.timestamp.asc()).all()
        
        # Format logs for the React Terminal
        logs = [{"agent": s.agent_name, "message": s.thought} for s in steps]
        
        # Pull the final_output from the last step (Judge) if completed
        final_result = None
        if exp.status == "COMPLETED" and steps:
            # We look for the 'final_output' key inside the JSONB details column
            final_result = steps[-1].details.get("final_output")

        return {
            "status": exp.status,
            "logs": logs,
            "final_result": final_result
        }
    finally:
        db.close()