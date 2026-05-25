import os
from datetime import datetime

class PromptLogger:
    def __init__(self, mission_name: str, iteration: int, agent_name: str, agent_model: str):
        log_dir = os.path.join("results", "prompt_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        safe_agent_name = "".join(c for c in agent_name if c.isalnum() or c in (' ', '_')).rstrip()
        safe_model_name = agent_model.replace('/', '-').replace('\\', '-')
        
        filename = f"{mission_name}_iter{iteration}_{safe_agent_name}_{safe_model_name}.txt"
        self.filepath = os.path.join(log_dir, filename)

    def log(self, prompt: str, response: str):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"""
============================================================
TIMESTAMP: {timestamp}
============================================================

PROMPT:
------------------------------------------------------------
{prompt}
------------------------------------------------------------

RESPONSE:
------------------------------------------------------------
{response}
============================================================
\n
"""
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Warning: Could not write to prompt log file {self.filepath}. Error: {e}")