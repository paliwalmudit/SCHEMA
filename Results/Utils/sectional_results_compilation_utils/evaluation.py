import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import json
from eval_logger import log_results
import unicodedata
import re
import numpy as np

# Load once (important)
_device = "cuda" if torch.cuda.is_available() else "cpu"
_embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=_device
)

print(_device, "device used for evaluation")


reference_report = None
final_report = ""
conversion_eval = {}
paper_in_text = ""

empty_report = {
    "Name": "no proposal generated",
    "Title": "no proposal generated",
    "Short Hypothesis": "no proposal generated",
    "Related Work": "no proposal generated",
    "Abstract": "no proposal generated",
    "Experiments": "no proposal generated",
    "Risk Factors and Limitations": "no proposal generated",
    "Full_Report": "no proposal generated",
    # "Novelty": "no proposal generated"
}

def parse_json_string(text):
    
    # Add every field name here
    keys = [
        "Name",
        "Title",
        "Short Hypothesis",
        "Related Work",
        "Abstract",
        "Experiments",
        "Risk Factors and Limitations"
    ]

    # Build regex to locate each key
    pattern = r'(' + '|'.join(map(re.escape, keys)) + r'):'

    matches = list(re.finditer(pattern, text))

    result = {}

    for i, match in enumerate(matches):
        key = match.group(0)  # e.g. "Title:"
        key = key[:-1]  # Remove the colon
        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        value = text[start:end].strip()
        result[key] = value

    # print(result)
    return result

def clean_json_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)      # Normalize fancy quotes/hyphens
    t = re.sub(r'\s+', ' ', t)               # Collapse all whitespace to single space
    return t.strip() 


def read_ref(ref_report):
    global paper_in_text
    path = os.path.join("Reference_Reports_json", f"{ref_report}.json")
    with open(path, 'r', encoding="utf-8") as file:
        data = json.load(file)
    paper_in_text = data["PAPER"]
    
    ref_report_json_data = parse_json_string(data["JSON"])
    ref_report_json_data["Full_Report"] = data["JSON"]
    # ref_report_json_data["Novelty"] = data["Novelty"] # Only if novety is present in the reference report, otherwise it will be empty

    # return data["JSON"]
    return ref_report_json_data


def structured_json_to_text(data):

    final_data = {}

    # handle dict directly
    if isinstance(data, dict):
        parsed = data
    elif isinstance(data, str):
        parsed = json.loads(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    lines = []
    for key, value in parsed.items():
        # print(key, type(value))
        # print(key)
        if isinstance(value, list):
            lines.append(f"{key}: {",".join([str(item) for item in value])}")
            final_data[key] = ",".join([str(item) for item in value])
        else:
            # print(key)
            final_data[key] = clean_json_text(value)
            lines.append(f"{key}: {value}")
    keys=[]
    for key, value in final_data.items():
        keys.append(key)
    report_in_text = clean_json_text("\n".join(lines))
    if "Experiments" not in keys:
        print(final_data, sep="\n-------------------\n")
        raise ValueError("Experiments field is missing in the final report JSON data.")
    final_data["Full_Report"] = report_in_text

    # return report_in_text
    return final_data
    
# there will be one name, and that name will be of the file(both ref report and final report should have same name)
def start_eval(fin_report, gen_from = "VL"):
    # get the reference report into the same format as the final_report, now to make it work with dataset, 
    # make it like it take text from the dataset and use the text directly and convert it into json idea, no need of pdf reading.
    global reference_report
    global final_report
    global conversion_eval, paper_in_text
    ref_report = fin_report
    
    reference_report = read_ref(ref_report)    

    path = os.path.join("Json_Results", f"{fin_report}.json")
    with open(path, 'r', encoding="utf-8") as file:
        data = json.load(file)
        if gen_from == "AS":
            final_report = structured_json_to_text(data[0]) if len(data)>0 else empty_report
        else:
            final_report = structured_json_to_text(data)
    conversion_eval["Generated_proposal_and_the_paper"] = evaluate_report(final_report["Full_Report"], paper_in_text)
    try:
        result1 = evaluate_report(final_report["Full_Report"], reference_report["Full_Report"])
        result1["Generated_proposal_and_ref_proposal"] = {}
        for key,value in final_report.items():
            if key not in ["Full_Report", "Novelty"]:
                result1["Generated_proposal_and_ref_proposal"][key] = evaluate_report(final_report[key], reference_report[key])
            if key == "Novelty":
                for novelty_key, novelty_value in final_report[key].items():
                    result1["Generated_proposal_and_ref_proposal"][key][novelty_key] = evaluate_report(final_report[key][novelty_key], reference_report[key][novelty_key])
        result2 = compare_reports(final_report["Full_Report"], reference_report["Full_Report"])
        result2["Sec_Generated_proposal_and_ref_proposal"] = {}
        for key,value in final_report.items():
            if key not in ["Full_Report", "Novelty"]:
                result2["Sec_Generated_proposal_and_ref_proposal"][key] = compare_reports(final_report[key], reference_report[key])
        try:
            log_results(result1, result2, conversion_eval, gen_from, run_name=f"eval_{ref_report}")
        except Exception as e:
            print("Error in logging results:", e)
    except Exception as e:
        print("Error in evaluating report:", e)
 

def evaluate_report(converted, reference):
    # evalauate the final report against the reference report, and also store the results somewhere
    # global reference_report
    # global final_report
    texts = [
        converted,
        reference
    ]

    # now do embedding based eval
    # Encode both reports
    embeddings = _embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    distance = np.linalg.norm(embeddings[0] - embeddings[1])

    cosine_simi = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1)  
    )[0][0]

    euclidean_distance = np.linalg.norm(
    embeddings[0] - embeddings[1]
    )

    # all results
    results = {
        "Embedding Cosine Similarity": float(cosine_simi),
        "Embedding Euclidean Distance": float(euclidean_distance),
        "Embedding Dim": embeddings.shape[1],
        "Embedding Model": "all-MiniLM-L6-v2"
    }

    return results

# For other evaluations [it includes embedding based evals and also cosine sentance embedding also includes that too, if not needed then can be removed]

def _simple_tokenize(text):
    # lower-case, split on non-word, filter empties
    return [t for t in re.split(r"\W+", text.lower()) if t]


def _try_import(module_name, package_name=None):
    try:
        module = __import__(module_name)
        return module
    except Exception:
        # try alternative package name (e.g., 'rouge_score')
        if package_name:
            try:
                module = __import__(package_name)
                return module
            except Exception:
                pass
        return None

def bleu_score(candidate, reference):
    """
    Sentence-level BLEU using sacrebleu if available, else nltk.
    Returns float in [0,1].
    """
    sacrebleu = _try_import("sacrebleu")
    if sacrebleu:
        try:
            # sacrebleu expects list of references (each a list for multi-ref)
            score = sacrebleu.corpus_bleu([candidate], [[reference]]).score / 100.0
            return score
        except Exception:
            pass

    # fallback to nltk
    nltk = _try_import("nltk")
    if nltk:
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            c_tok = _simple_tokenize(candidate)
            r_tok = _simple_tokenize(reference)
            if not c_tok or not r_tok:
                return 0.0
            smoothie = SmoothingFunction().method4
            return sentence_bleu([r_tok], c_tok, smoothing_function=smoothie)
        except Exception:
            pass

    raise RuntimeError("No BLEU backend available. Install 'sacrebleu' or 'nltk'.")

def rouge_scores(candidate, reference):
    """
    Returns dict of rouge-1/2/l with precision, recall, fmeasure if rouge_score package is available.
    """
    rs = _try_import("rouge_score", "rouge_score")
    if rs:
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
            scores = scorer.score(reference, candidate)  # note: scorer takes (target, prediction)
            # Convert to simple dict
            out = {}
            for k, v in scores.items():
                out[k] = {"precision": v.precision, "recall": v.recall, "fmeasure": v.fmeasure}
            return out
        except Exception as e:
            raise RuntimeError(f"Error running rouge_score: {e}")
    raise RuntimeError("rouge_score not installed. pip install rouge-score")

def bertscore_scores(candidates, references, model_type="microsoft/deberta-xlarge-mnli", device=None, rescale_with_baseline=True, lang = "en"):
    """
    Returns dict with lists or single P/R/F1 values (0..1).
    Uses bert_score.score if available.
    """
    bs = _try_import("bert_score")
    if not bs:
        raise RuntimeError("bert-score not installed. pip install bert-score")
    try:
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    try:
        P, R, F1 = bs.score(candidates, references, model_type=model_type, lang = lang, rescale_with_baseline=rescale_with_baseline, device=device)
        # convert tensors to floats or lists
        def _to_floats(t):
            try:
                return [float(x) for x in t]
            except Exception:
                return float(t.mean().item())
        return {"precision": _to_floats(P), "recall": _to_floats(R), "f1": _to_floats(F1)}
    except Exception as e:
        raise RuntimeError(f"bert-score failed: {e}")

def compare_reports(final_rep, ref_rep, metrics=("bleu", "rouge", "bertscore"), 
                    bert_model="microsoft/deberta-xlarge-mnli",
                    device=None):
    """
    Compare two reports (strings). Return dict with metric results and any warnings.
    - metrics: iterable of metric names to compute.
    - bert_model: model identifier for bert-score (default: DeBERTa xlarge for better quality; change if you lack resources)
    we can use microsoft/deberta-xlarge-mnli [best but resource intensive], roberta-large-mnli [good], distilroberta-base[decent] or bert-base-uncased [faster]
    - device: "cuda" or "cpu" or None (auto)
    """
    global reference_report
    global final_report
    metrics = [m.lower() for m in metrics]
    result = {"metrics_requested": metrics, "warnings": [], "values": {}}

    report_a = final_rep
    report_b = ref_rep
    
    # BLEU
    if "bleu" in metrics:
        try:
            # BLEU expects token lists but our wrapper accepts strings and handles tokenization
            result["values"]["bleu"] = bleu_score(report_a, report_b)
        except Exception as e:
            result["values"]["bleu"] = None
            result["warnings"].append(f"bleu error: {e}")

    # ROUGE
    if "rouge" in metrics:
        try:
            result["values"]["rouge"] = rouge_scores(report_a, report_b)
        except Exception as e:
            result["values"]["rouge"] = None
            result["warnings"].append(f"rouge error: {e}")

    # BERTScore
    if "bertscore" in metrics:
        try:
            # bertscore.score accepts lists of strings
            bs = bertscore_scores([report_a], [report_b], model_type=bert_model, device=device)
            # if lists, collapse to mean
            def _collapse(x):
                if isinstance(x, list):
                    return float(sum(x) / len(x)) if x else None
                return x
            result["values"]["bertscore"] = {"precision": _collapse(bs["precision"]),
                                            "recall": _collapse(bs["recall"]),
                                            "f1": _collapse(bs["f1"])}
        except Exception as e:
            result["values"]["bertscore"] = None
            result["warnings"].append(f"bertscore error: {e}")


    # clean warnings: if empty, remove field
    if not result["warnings"]:
        result.pop("warnings")

    return result


# if __name__ == "__main__":
#     start_eval("agent_drift")

# === Example usage ===

# if __name__ == "__main__":
#     ref_path = os.path.join("Reference_Reports", f"agent_drift.pdf")
#     print(read_pdf("agent_drift"))
    
# From start_eval
   
    # df1 = pd.DataFrame([result1])
    # df2 = pd.json_normalize(result2)

    # print(df1, df2, sep="\n\n")

if __name__ == "__main__":
    files = [f for f in os.listdir("Full_dataset_eval") if f.endswith(".json")]
    for file in files:
        file_name = os.path.splitext(file)[0]
        print(f"Evaluating {file_name}...")
        start_eval(file_name, gen_from="VL")