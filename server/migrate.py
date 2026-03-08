"""
One-time migration: drops and recreates all tables with the updated schema.
Run once from the server/ directory, then delete this file.
"""
import sys
sys.path.insert(0, ".")

from database.session import engine
from database.base import Base
from database.models import Experiment, AgentStep  # ensure models are registered

print("Dropping existing tables...")
Base.metadata.drop_all(bind=engine)

print("Recreating tables with new schema...")
Base.metadata.create_all(bind=engine)

print("Done! Tables recreated successfully.")
