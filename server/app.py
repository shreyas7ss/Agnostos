"""
these are the routes for the server i have created
"""

from cv2.dnn import Target
from fastapi import FastAPI, BackgroundTasks, HTTPException
from typing import List
from pydantic import BaseModel

from graph.workflow import agnostos_graph
from database.session import SessionLocal
from database.models import Experiment, AgentStep
from utils.config import settings

app = FastAPI(title="Agnostos Lab API", version="1.0.0")

class ExperimentRequest(BaseModel):
    dataset_path: str
    target_column: str


async def run_agnostas_task(experiment_id: str, dataset_path: str):
    db = SessionLocal()
    try:
        initial_state={
            "experiment_id":experiment_id,
            "dataset_path":dataset_path,
            "data_manifesto":{"target_column":"target"},
            "messages":[]
        }
    
    async for output in agnostos_graph.astream(initial_state):
        for node_name,state_update in output.items():
            
            new_step = AgentStep(
                experiment_id=experiment_id,
                agent_name=node_name,
                thought=str(state_update.get("messages",["Node finished"])[-1].content),
                detailes=state_update
            )

        db.add(new_step)
        db.commit()

        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        experiment.status = "COMPLETED"
        db.commit()

    except Exception as e:
        print(f"Error in background task: {e}")
    finally:
        db.close()


#routes 

@app.post("/experiment/start")    
async def start_experiment(req: ExperimentRequest, background_tasks: BackgroundTasks):
    """Starts a new Parallel ML Experiment."""
    db = SessionLocal()
    
    # 1. Create entry in Neon
    new_exp = Experiment(status="RUNNING", dataset_path=req.dataset_path)
    db.add(new_exp)
    db.commit()
    db.refresh(new_exp)
    
    # 2. Fire and Forget the LangGraph
    background_tasks.add_task(run_agnostos_task, new_exp.id, req.dataset_path)
    
    return {"experiment_id": new_exp.id, "status": "Started"}

@app.get("/experiments/{exp_id}")
async def get_status(exp_id: int):
    """Fetches the current status and logs of an experiment."""
    db = SessionLocal()
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Not found")
    
    # Join with steps to show the "Terminal" view in the UI
    steps = db.query(AgentStep).filter(AgentStep.experiment_id == exp_id).all()
    
    return {
        "status": exp.status,
        "logs": [{"agent": s.agent_name, "message": s.thought} for s in steps],
        "final_result": steps[-1].details.get("final_output") if exp.status == "COMPLETED" else None
    }
