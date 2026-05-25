import os
import json


def save_paper(result_dict: dict, run_name: str = "eval"):
    """
    Stores evaluation results into a single file under results/eval_results.
    """

    # Directory setup
    log_dir = "Reference_Reports_json"
    os.makedirs(log_dir, exist_ok=True)

    # File naming
    filename = f"{run_name}.json"
    filepath = os.path.join(log_dir, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
            print(f"Conversion Completed and stored at:{filepath}")
    except Exception as e:
        print(f"Warning: Could not write evaluation results to {filepath}. Error: {e}")

