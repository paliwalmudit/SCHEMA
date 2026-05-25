import os
import json
from datetime import datetime


def log_results(result_dict: dict, extra_results_dict: dict, conversion_eval:dict, gen_name, run_name: str = "eval"):
    """
    Stores evaluation results into a single file under results/eval_results.
    """

    # Directory setup
    log_dir = os.path.join("results", "eval_results")
    os.makedirs(log_dir, exist_ok=True)

    # Timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{run_name}_{gen_name}.json"
    filepath = os.path.join(log_dir, filename)

    # Combine payload (keep structure explicit)
    payload = {
        "timestamp": timestamp,
        "Generated_By": ["Virtual Lab" if gen_name == "VL" else "AI-Scientist"][0],
        "Conversion_results": conversion_eval,
        "primary_results": result_dict,
        "secondary_results": extra_results_dict
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"Evaluation Completed and stored at:{filepath}")
    except Exception as e:
        print(f"Warning: Could not write evaluation results to {filepath}. Error: {e}")
