from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import ForeignKey, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB
from .base import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id:           Mapped[int]           = mapped_column(Integer, primary_key=True)
    name:         Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status:       Mapped[str]           = mapped_column(String(50),  nullable=False, default="RUNNING")
    dataset_path: Mapped[str]           = mapped_column(String(512), nullable=False)
    created_at:   Mapped[datetime]      = mapped_column(DateTime, default=datetime.utcnow)
    manifest:     Mapped[Optional[dict]]= mapped_column(JSONB,    nullable=True)

    steps: Mapped[List["AgentStep"]] = relationship(back_populates="experiment")


class AgentStep(Base):
    __tablename__ = "agent_steps"

    id:            Mapped[int]                  = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int]                  = mapped_column(ForeignKey("experiments.id"))
    agent_name:    Mapped[str]                  = mapped_column(String(50), nullable=False)
    thought:       Mapped[str]
    details:       Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    timestamp:     Mapped[datetime]             = mapped_column(DateTime, default=datetime.utcnow)

    experiment: Mapped["Experiment"] = relationship(back_populates="steps")