import json
from pathlib import Path
import argparse

# ======================================================
# Configuration
# ======================================================

def main(gen_name):
    folder_a = Path(f"results\\reviews_evaluated\\{gen_name}\\gpt-5.6-sol")
    folder_b = Path(f"results\\reviews_evaluated\\{gen_name}\\gemini-3.1-pro")


    output_folder = Path(f"results\\final_eval_GPT_gemini\\{gen_name}") 
    output_folder.mkdir(parents=True, exist_ok=True)

    # ======================================================
    # Main
    # ======================================================
    count = 0
    for file_a in folder_a.glob("*.json"):

        file_b = folder_b / file_a.name
        count+=1
        if not file_b.exists():
            print(f"Skipping {file_a.name} (missing in parallel-eval folder)")
            continue

        with open(file_a, "r", encoding="utf-8") as f:
            data_a = json.load(f)

        with open(file_b, "r", encoding="utf-8") as f:
            data_b = json.load(f)

        output = {}

        for criterion in data_a:
            score_a = data_a[criterion]["score"]
            score_b = data_b[criterion]["score"]

            output[criterion] = {
                "score": round((score_a + score_b) / 2, 2)
            }

        out_file = output_folder / file_a.name

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)

    print(count)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_name", required=True)
    args = parser.parse_args()
    main(args.gen_name)