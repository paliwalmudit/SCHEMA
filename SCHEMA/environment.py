import os
import json

MISSIONS_FOLDER = "INPUT"

def load_mission_file(research_topic: str) -> dict:
    file_path = os.path.join(
        MISSIONS_FOLDER,
        f"{research_topic}.json"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mission file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

class MissionEnvironment:
    def __init__(self, mission_name: str):
        
        try:
            plan = load_mission_file(mission_name)
        except Exception as e:
            raise ValueError("FILE NOT FOUND", e)
        keywords = ", ".join(plan["keywords"])
        self.topic = mission_name
        self.title = plan["title"]
        self.desc=plan["title"]+" ,Core idea is: "+plan["core_idea"]+" ,keywords to are: "+keywords+" ,other information:"+plan["tldr"]
        self.duration = 3 # number of meetings
        self.crew_size = 6 # number of agents
# 3M3A
# For automated testing pipeline instead of using mission plan just use the json file for mission plan (dataset) and have the mission name as field_{SerialNumber}

