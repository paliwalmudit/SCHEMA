import json
from llm_api import get_llm_response

Comparision_prompt = """
You are given two pieces of data: the original data and the converted data.
Your task is to determine if base idea conversion from the original data to the converted data is correct or not.
You do not have to check for exact match, just the base idea conversion.
original data: {original}
converted data: {converted}
return the output as a boolean value: True if the conversion is correct, False otherwise.
"""

def check (original, converted):
    """
    Check if the conversion from original to converted is correct.
    This is a placeholder function and should be implemented with actual logic.
    """
    # Placeholder logic: check if the converted string contains the original string
    data = json.loads(converted)
    query = Comparision_prompt.format(original=original, converted=data)
    response = get_llm_response("gpt-5-nano", query, temperature=0)
    return response.strip().lower() == "true"

# Example usage
# print(check("world simulation was created using multiple agents", '''{"title": "MultiAgent Simulation", "Text": "in this research we will used multi agent system to simulate world"}'''))