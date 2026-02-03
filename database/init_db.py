from .session import engine
from .base import Base
from .models import Experiment, AgentStep


def init_db():
   
   try:
     Base.metadata.create_all(bind=engine)
     print("succesfully created database")

   except Exception as e:
    print(f"An error occurred: {e}")

if __name__ == "__main__":
    init_db()
    