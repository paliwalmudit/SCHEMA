"""
Compare GEN_AI_SCI vs GEN_VL scores, category by category.

- Files are matched by filename between the two folders
  (e.g. "Engineering & Computer Science_1.json" in GEN_AI_SCI
   is paired with the file of the same name in GEN_VL).
- For each category (Consistency, Clarity, Novelty, ...), runs a
  paired t-test (scipy.stats.ttest_rel) across all matched files.
- Reports Cohen's d for paired samples (dz = mean(diff) / std(diff)),
  which is the standard effect size to report alongside a paired t-test.
- Runs a Shapiro-Wilk test on the paired differences (the actual
  assumption behind a paired t-test is that the DIFFERENCES are
  normal, not the raw scores) and reports whether normality is
  rejected. With N ~ 91 the t-test is fairly robust to mild non-
  normality via the CLT, but it's good practice to check.
- Flags each category as "Significant" (p < ALPHA) or not.

Usage:
    python compare_gen_ai_sci_vs_gen_vl.py

Edit BASE_DIR below to point at your "final_eval_GPT_gemini" folder.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# CONFIG - edit these paths to match your setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(".")
DIR_A = BASE_DIR / "GEN_AI_SCI"
DIR_B = BASE_DIR / "GEN_VL"

OUTPUT_CSV = "gen_ai_sci_vs_gen_vl_results.csv"
ALPHA = 0.05  # significance threshold for both the t-test and the Shapiro-Wilk check


def load_scores(json_path: Path) -> dict:
    """Load a json file and return {category: score}."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {cat: vals["score"] for cat, vals in data.items() if "score" in vals}


def cohen_d_paired(diffs: np.ndarray) -> float:
    """Cohen's d_z for paired/repeated-measures data: mean(diff) / std(diff, ddof=1)."""
    diffs = np.asarray(diffs, dtype=float)
    sd = diffs.std(ddof=1)
    if sd == 0:
        return np.nan
    return diffs.mean() / sd


def main():
    files_a = {p.name: p for p in DIR_A.glob("*.json")}
    files_b = {p.name: p for p in DIR_B.glob("*.json")}

    common = sorted(set(files_a) & set(files_b))
    missing_a = sorted(set(files_b) - set(files_a))
    missing_b = sorted(set(files_a) - set(files_b))

    if missing_a:
        print(f"[warn] Present in GEN_VL but not GEN_AI_SCI: {missing_a}")
    if missing_b:
        print(f"[warn] Present in GEN_AI_SCI but not GEN_VL: {missing_b}")
    if not common:
        raise SystemExit("No matching filenames found between the two folders.")

    # Build a long-form table: filename, category, score_A, score_B
    rows = []
    for fname in common:
        scores_a = load_scores(files_a[fname])
        scores_b = load_scores(files_b[fname])
        categories = set(scores_a) & set(scores_b)
        for cat in categories:
            rows.append({
                "file": fname,
                "category": cat,
                "score_A_GEN_AI_SCI": scores_a[cat],
                "score_B_GEN_VL": scores_b[cat],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No overlapping categories found across matched files.")

    results = []
    for cat, group in df.groupby("category"):
        a = group["score_A_GEN_AI_SCI"].to_numpy(dtype=float)
        b = group["score_B_GEN_VL"].to_numpy(dtype=float)
        n = len(a)

        if n < 2:
            print(f"[warn] Category '{cat}' has only {n} paired sample(s), skipping stats.")
            continue

        t_stat, p_val = stats.ttest_rel(a, b)
        diffs = a - b
        d = cohen_d_paired(diffs)

        # Shapiro-Wilk needs at least 3 points; the assumption that matters
        # for a paired t-test is normality of the DIFFERENCES, not the raw scores.
        if n >= 3:
            sw_stat, sw_p = stats.shapiro(diffs)
            normal_diffs = sw_p >= ALPHA  # fail to reject H0 of normality
        else:
            sw_stat, sw_p, normal_diffs = np.nan, np.nan, np.nan

        # Wilcoxon signed-rank test: non-parametric counterpart to the paired
        # t-test, doesn't assume normality of the differences. Useful as a
        # robustness check, especially when diffs_normal comes back False.
        # Requires at least one non-zero difference.
        nonzero_diffs = diffs[diffs != 0]
        if len(nonzero_diffs) >= 1:
            try:
                w_stat, w_p = stats.wilcoxon(diffs)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
        else:
            w_stat, w_p = np.nan, np.nan  # all differences are exactly zero

        results.append({
            "category": cat,
            "n_pairs": n,
            "mean_GEN_AI_SCI": a.mean(),
            "mean_GEN_VL": b.mean(),
            "mean_diff (A-B)": diffs.mean(),
            "t_stat": t_stat,
            "p_value": p_val,
            "significant (p<%.2f)" % ALPHA: bool(p_val < ALPHA),
            "cohens_d_paired": d,
            "shapiro_p (diffs)": sw_p,
            "diffs_normal (p>=%.2f)" % ALPHA: normal_diffs,
            "wilcoxon_p": w_p,
            "wilcoxon_significant (p<%.2f)" % ALPHA: (bool(w_p < ALPHA) if not np.isnan(w_p) else np.nan),
        })

    results_df = pd.DataFrame(results).sort_values("category").reset_index(drop=True)

    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")
    print("\n=== GEN_AI_SCI vs GEN_VL — paired t-test & Cohen's d per category ===\n")
    print(results_df.to_string(index=False))

    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

