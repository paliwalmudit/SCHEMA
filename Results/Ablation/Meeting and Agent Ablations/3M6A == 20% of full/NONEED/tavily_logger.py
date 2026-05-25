import os
import json
from datetime import datetime

def log_tavily(query: str, response: str, number: int, project_name: str = "default_project"):
    """
    Logs Tavily search queries and responses into a JSON file.
    It keeps appending new entries to the same file for each run.
    """

    folder_name = os.path.join("results", "Tavily_Search_Logs")
    os.makedirs(folder_name, exist_ok=True)

    file_path = os.path.join(folder_name, f"{project_name}_{number}_tavily_log.json")

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "response": response
    }

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(log_entry)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # print("\n--- Tavily Log Saved ---")
    # print(f"Log file updated: {file_path}")
