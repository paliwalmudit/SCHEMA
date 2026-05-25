from main import AVAILABLE_MODELS
from agents import PIAgent
from prompts import PI_MODEL_SELECTION_PROMPT
from llm_api import get_llm_response
import json    

def interview(PI: PIAgent, agent_information:dict):
    # Here i have to setup an interview enviroment where PI will generate N number of agents based on the research topic and the agent information provided. 
    # Then PI will interview each agent based on their skill set and capabilities required for this task
    # Then PI will select some of the best agents it thinks are suitable for this research topic.
    pass

    
def HireorFire(PI: PIAgent, agent_information:dict):
    pass