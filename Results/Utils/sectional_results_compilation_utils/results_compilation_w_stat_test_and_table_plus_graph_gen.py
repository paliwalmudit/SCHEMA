import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.stats import ttest_ind
from scipy.stats import ttest_rel

# ================= CONFIG =================

BASE_DIR = r"results"
OUTPUT_DIR = "FIGURES_Combined"

DATASETS = {
    "AI-SCI": os.path.join(BASE_DIR, "Output_AI-SCI", "results", "AS"),
    "VL": os.path.join(BASE_DIR, "Output_VL", "results", "VL")
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= METRICS =================

METRICS = [
    {
        "label": "Similarity with Proposal",
        "extract": lambda data: data["primary_results"]["Embedding Cosine Similarity"],
        "type": "higher"
    },
    {
        "label": "Distance with Proposal",
        "extract": lambda data: data["primary_results"]["Embedding Euclidean Distance"],
        "type": "lower"
    },
    {
        "label": "Similarity with Paper",
        "extract": lambda data: data["Conversion_results"]["Generated_proposal_and_the_paper"]["Embedding Cosine Similarity"],
        "type": "higher"
    },
    {
        "label": "Distance with Paper",
        "extract": lambda data: data["Conversion_results"]["Generated_proposal_and_the_paper"]["Embedding Euclidean Distance"],
        "type": "lower"
    },
    {
        "label": "BERTScore (F1)",
        "extract": lambda data: data["secondary_results"]["values"]["bertscore"]["f1"],
        "type": "higher"
    },
]

# ================= HELPERS =================

def get_section_scores(data):
    try:
        sections = data["primary_results"]["Generated_proposal_and_ref_proposal"]
        return {
            section: {
                "Similarity with Proposal": values["Embedding Cosine Similarity"],
                "Distance with Proposal": values["Embedding Euclidean Distance"]
            }
            for section, values in sections.items()
            if isinstance(values, dict)
            and "Embedding Cosine Similarity" in values
            and "Embedding Euclidean Distance" in values
        }
    except Exception:
        return {}


def safe_extract(metric_extract_fn, data):
    try:
        val = metric_extract_fn(data)
        if val is None:
            return np.nan
        return float(val)
    except Exception:
        return np.nan


def aggregate_scores(score_list):
    arr = np.array(score_list, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    return arr.mean(), arr.std()

# paired t-test
def calculate_significance(list1, list2):
    """Calculates Paired t-test returning (p_value, is_significant)"""
    # Ensure lists are the same length for pairing
    min_len = min(len(list1), len(list2))
    l1 = list1[:min_len]
    l2 = list2[:min_len]
    
    # Filter out pairs where either value is NaN
    valid_l1, valid_l2 = [], []
    for x, y in zip(l1, l2):
        if not np.isnan(x) and not np.isnan(y):
            valid_l1.append(x)
            valid_l2.append(y)

    if len(valid_l1) > 1 and len(valid_l2) > 1:
        # ttest_rel is the Paired t-test
        stat, p = ttest_rel(valid_l1, valid_l2)
        return p, p < 0.05
    return np.nan, False

# welch t-test (unpaired)
# def calculate_significance(list1, list2):
#     """Calculates Welch's t-test returning (p_value, is_significant)"""
#     l1 = [x for x in list1 if not np.isnan(x)]
#     l2 = [x for x in list2 if not np.isnan(x)]
#     if len(l1) > 1 and len(l2) > 1:
#         stat, p = ttest_ind(l1, l2, equal_var=False)
#         return p, p < 0.05
#     return np.nan, False


def process_directory(dir_path):
    overall_scores = {metric["label"]: [] for metric in METRICS}
    section_scores = {}

    if not os.path.exists(dir_path):
        print(f"Warning: Directory not found -> {dir_path}")
        return overall_scores, section_scores

    for file in sorted(os.listdir(dir_path)):
        if not file.endswith(".json"):
            continue

        file_path = os.path.join(dir_path, file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # Extract Overall
        for metric in METRICS:
            score = safe_extract(metric["extract"], data)
            overall_scores[metric["label"]].append(score)

        # Extract Section
        current_sections = get_section_scores(data)
        for section, metrics in current_sections.items():
            if section not in section_scores:
                section_scores[section] = {
                    "Similarity with Proposal": [],
                    "Distance with Proposal": []
                }
            section_scores[section]["Similarity with Proposal"].append(metrics["Similarity with Proposal"])
            section_scores[section]["Distance with Proposal"].append(metrics["Distance with Proposal"])

    return overall_scores, section_scores


# ==========================================================
# PROCESSING DATASETS
# ==========================================================

processed_data = {}

for dataset_name, path in DATASETS.items():
    raw_overall, raw_section = process_directory(path)
    
    # Aggregate overall
    agg_overall = {}
    for metric in METRICS:
        lbl = metric["label"]
        mean, std = aggregate_scores(raw_overall[lbl])
        agg_overall[lbl] = (mean, std)
        
    # Aggregate section
    agg_section = {}
    for sec, vals in raw_section.items():
        sim_m, sim_s = aggregate_scores(vals["Similarity with Proposal"])
        dist_m, dist_s = aggregate_scores(vals["Distance with Proposal"])
        agg_section[sec] = {
            "sim_mean": sim_m, "sim_std": sim_s,
            "dist_mean": dist_m, "dist_std": dist_s
        }
        
    processed_data[dataset_name] = {
        "overall_raw": raw_overall,
        "section_raw": raw_section,
        "overall": agg_overall,
        "section": agg_section
    }

# ==========================================================
# OVERALL TABLE DATA
# ==========================================================

overall_rows = []
dataset_keys = list(DATASETS.keys())
ds1, ds2 = dataset_keys[0], dataset_keys[1]

for metric in METRICS:
    lbl = metric["label"]
    row = {"Metric": lbl}
    
    for ds in dataset_keys:
        mean, std = processed_data[ds]["overall"][lbl]
        row[ds] = f"{mean:.3f} ± {std:.3f}" if not np.isnan(mean) else "NA"
        
    p_val, sig = calculate_significance(
        processed_data[ds1]["overall_raw"][lbl], 
        processed_data[ds2]["overall_raw"][lbl]
    )
    row["P-Value"] = f"{p_val:.3e}" if not np.isnan(p_val) else "NA"
    row["Sig (<0.05)?"] = "Yes" if sig else "No"
    
    overall_rows.append(row)

overall_df = pd.DataFrame(overall_rows)

# ==========================================================
# SECTION TABLE DATA
# ==========================================================

all_sections = sorted(list(set(
    sec for ds in processed_data.values() for sec in ds["section"].keys()
)), key=lambda x: x.lower())

section_rows = []
for sec in all_sections:
    row = {"Section": sec}
    
    # Similarity Columns
    for ds in dataset_keys:
        m = processed_data[ds]["section"].get(sec, {}).get("sim_mean", np.nan)
        s = processed_data[ds]["section"].get(sec, {}).get("sim_std", np.nan)
        row[f"{ds} Sim"] = f"{m:.3f} ± {s:.3f}" if not np.isnan(m) else "NA"
        
    p_val_sim, sig_sim = calculate_significance(
        processed_data[ds1]["section_raw"].get(sec, {}).get("Similarity with Proposal", []),
        processed_data[ds2]["section_raw"].get(sec, {}).get("Similarity with Proposal", [])
    )
    row["Sim P-Val"] = f"{p_val_sim:.3e}" if not np.isnan(p_val_sim) else "NA"
    row["Sim Sig?"] = "Yes" if sig_sim else "No"
    
    # Distance Columns
    for ds in dataset_keys:
        m = processed_data[ds]["section"].get(sec, {}).get("dist_mean", np.nan)
        s = processed_data[ds]["section"].get(sec, {}).get("dist_std", np.nan)
        row[f"{ds} Dist"] = f"{m:.3f} ± {s:.3f}" if not np.isnan(m) else "NA"

    p_val_dist, sig_dist = calculate_significance(
        processed_data[ds1]["section_raw"].get(sec, {}).get("Distance with Proposal", []),
        processed_data[ds2]["section_raw"].get(sec, {}).get("Distance with Proposal", [])
    )
    row["Dist P-Val"] = f"{p_val_dist:.3e}" if not np.isnan(p_val_dist) else "NA"
    row["Dist Sig?"] = "Yes" if sig_dist else "No"
        
    section_rows.append(row)

section_df = pd.DataFrame(section_rows)

# ==========================================================
# PRINT RESULTS TO CONSOLE
# ==========================================================
print("\n" + "="*60)
print("OVERALL METRICS COMPARISON (WITH SIGNIFICANCE)")
print("="*60)
print(overall_df.to_string(index=False))

print("\n" + "="*120)
print("SECTION METRICS COMPARISON (WITH SIGNIFICANCE)")
print("="*120)
print(section_df.to_string(index=False))
print("\n")

# ==========================================================
# COMBINED FIGURE OUTPUT (TABLES WITH ROW-BY-ROW BOLDING)
# ==========================================================

# Increased figure width to accommodate the new significance columns
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 3])

# --- 1. OVERALL TABLE ---
ax1 = fig.add_subplot(gs[0])
ax1.axis("off")
table1 = ax1.table(
    cellText=overall_df.values,
    colLabels=overall_df.columns,
    cellLoc="center",
    loc="center"
)
table1.auto_set_font_size(False)
table1.set_fontsize(10)
table1.scale(1, 1.6)
ax1.set_title("Overall Metrics Comparison", pad=10, fontweight="bold")

# Bold winning dataset row-by-row
for row_idx, metric in enumerate(METRICS):
    m_type = metric["type"]
    lbl = metric["label"]
    
    val1 = processed_data[ds1]["overall"][lbl][0]
    val2 = processed_data[ds2]["overall"][lbl][0]
    
    if not np.isnan(val1) and not np.isnan(val2):
        if m_type == "higher":
            best_ds_idx = 0 if val1 > val2 else 1
        else:
            best_ds_idx = 0 if val1 < val2 else 1
            
        # +1 for header row offset, +1 because col 0 is 'Metric'
        table1[row_idx + 1, best_ds_idx + 1].get_text().set_fontweight("bold")

# --- 2. SECTION TABLE ---
ax2 = fig.add_subplot(gs[1])
ax2.axis("off")

if not section_df.empty:
    table2 = ax2.table(
        cellText=section_df.values,
        colLabels=section_df.columns,
        cellLoc="center",
        loc="center"
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.4)
    ax2.set_title("Section Metrics Comparison (Side-by-Side)", pad=10, fontweight="bold")

    for sec_idx, sec in enumerate(all_sections):
        row_table_idx = sec_idx + 1  # Header offset
        
        # 1. Similarity (Columns 1 and 2, higher is better)
        v1_sim = processed_data[ds1]["section"].get(sec, {}).get("sim_mean", np.nan)
        v2_sim = processed_data[ds2]["section"].get(sec, {}).get("sim_mean", np.nan)
        if not np.isnan(v1_sim) and not np.isnan(v2_sim):
            best_col = 1 if v1_sim > v2_sim else 2
            table2[row_table_idx, best_col].get_text().set_fontweight("bold")

        # 2. Distance (Columns 5 and 6, lower is better)
        v1_dist = processed_data[ds1]["section"].get(sec, {}).get("dist_mean", np.nan)
        v2_dist = processed_data[ds2]["section"].get(sec, {}).get("dist_mean", np.nan)
        if not np.isnan(v1_dist) and not np.isnan(v2_dist):
            best_col = 5 if v1_dist < v2_dist else 6
            table2[row_table_idx, best_col].get_text().set_fontweight("bold")

else:
    ax2.text(0.5, 0.5, "No section-level data found", ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "combined_table.png"), dpi=300, bbox_inches="tight")
plt.close()

# ==========================================================
# GRAPH GENERATION (Unchanged logic, just runs cleanly)
# ==========================================================

# --- Overall Bar Chart ---
plt.figure(figsize=(12, 6))
labels = [m["label"] for m in METRICS]
x = np.arange(len(labels))
width = 0.35

for i, dataset_name in enumerate(dataset_keys):
    means = [processed_data[dataset_name]["overall"][lbl][0] for lbl in labels]
    plt.bar(x + (i - 0.5) * width, means, width, label=dataset_name)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Overall Metrics Comparison")
plt.xticks(x, [l.replace(" ", "\n") for l in labels], rotation=0)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "overall_metrics.png"), dpi=300, bbox_inches="tight")
plt.close()

# --- Section Similarity Comparison Bar Chart ---
if all_sections:
    plt.figure(figsize=(14, 6))
    x_sec = np.arange(len(all_sections))
    
    for i, dataset_name in enumerate(dataset_keys):
        sim_means = [
            processed_data[dataset_name]["section"].get(sec, {}).get("sim_mean", 0)
            for sec in all_sections
        ]
        plt.bar(x_sec + (i - 0.5) * width, sim_means, width, label=f"{dataset_name} Similarity")

    plt.ylim(0, 1)
    plt.xticks(x_sec, all_sections, rotation=45, ha="right")
    plt.ylabel("Cosine Similarity Score")
    plt.title("Section Similarity Comparison Across Datasets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "section_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close()

print(f"Saved figures with row-wise selective bolding and stat testing to: {OUTPUT_DIR}")