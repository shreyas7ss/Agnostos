from sklearn.metrics._dist_metrics import DiceDistance
from cv2 import detail
from datetime import datetime
from typing import List,Optional
from sqlalchemy import ForeignKey,DateTime,Integer,String
from sqlalchemy.orm import Mapped, mapped_column,relationship
from .base import Base
from sqlalchemy.dialects.postgresql import JSONB


class Experiment(Base):
    __tablename__ = "experiments"

    #coloumns of the first DB
    id : Mapped[int] = mapped_column(Integer,primary_key=True)
    name : Mapped[str] = mapped_column(String(255),nullable=False)
    created_at : Mapped[datetime] = mapped_column(default=datetime.utcnow())
    manifest: Mapped[Optional[dict]] = mapped_column(JSONB)
    steps : Mapped[list["AgentStep"]] = relationship(back_populates="experiment")

    
    
class AgentStep(Base):
    __tablename__ = "agent_steps"

    id : Mapped[int] = mapped_column(Integer,primary_key=True)
    agent_name : Mapped[str] = mapped_column(String(50),nullable=False)
    thought : Mapped[str]
    details : Mapped[Optional[Dict[str,Any]]]=mapped_column(JSONB)
    timestamp : Mapped[datetime] = mapped_column(default=datetime.utcnow())

    experiment :Mapped["Experiment"] = relationship(back_populates="steps")
    

    