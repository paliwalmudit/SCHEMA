from datetime import datetime
import os

def log(prompt, log_name, iteration, agent_information, selected_model, reasoning):
    # Directory setup
    log_dir = os.path.join("results", "model_selection_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_selection_{log_name}_{iteration}.txt"
    file_path = os.path.join(log_dir, filename)

    # Log content
    log_content = f"Timestamp: {timestamp}\n"
    log_content += f"Prompt: {prompt}\n"
    log_content += f"Agent Information: {agent_information}\n"
    log_content += f"Selected Model: {selected_model}\n"
    log_content += f"Reasoning: {reasoning}\n"

    # Write to file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(log_content)