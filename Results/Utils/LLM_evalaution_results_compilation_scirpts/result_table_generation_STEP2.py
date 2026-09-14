import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ======================================================
# Configuration
# ======================================================

BASE_FOLDER = Path("results/final_eval_GPT_gemini")
Figure_Folder = Path("results/Figures")
Figure_Folder.mkdir(parents=True, exist_ok=True)
# ======================================================
# Collect statistics
# ======================================================

rows = []

for version_folder in sorted(BASE_FOLDER.iterdir()):

    if not version_folder.is_dir():
        continue

    totals = {}
    counts = {}

    for json_file in version_folder.glob("*.json"):

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for criterion, values in data.items():

            score = values["score"]

            totals[criterion] = totals.get(criterion, 0) + score
            counts[criterion] = counts.get(criterion, 0) + 1

    row = {"Version": version_folder.name}

    for criterion in totals:
        row[criterion] = round(totals[criterion] / counts[criterion], 3)

    rows.append(row)

# ======================================================
# Create dataframe
# ======================================================

df = pd.DataFrame(rows)

criteria = [c for c in df.columns if c != "Version"]

# Higher is better
df["Rank"] = (
    df["OverallAssessment"]
    .rank(method="dense", ascending=False)
    .astype(int)
)

df = df.sort_values("Rank").reset_index(drop=True)


# ======================================================
# Create table figure
# ======================================================

display_df = df.copy()

# Round values
for col in criteria :
    display_df[col] = display_df[col].map(lambda x: f"{x:.2f}")

display_df["Rank"] = display_df["Rank"].astype(int)

# Put Rank first
cols = ["Rank", "Version"] + criteria
display_df = display_df[cols]

fig_height = max(1.5, 0.55 * len(display_df) + 1)

fig, ax = plt.subplots(figsize=(14, fig_height))
ax.axis("off")

table = ax.table(
    cellText=display_df.values,
    colLabels=display_df.columns,
    cellLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)

# Header formatting
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")

plt.tight_layout()

plt.savefig(
    Figure_Folder / "summary_table_GPT_gemini.png",
    dpi=300,
    bbox_inches="tight"
)

# plt.savefig(
#     Figure_Folder / "summary_table.pdf",
#     bbox_inches="tight"
# )

plt.close()

print(f"Saved summary table to {Figure_Folder / 'summary_table_GPT_gemini.png'}")

# # ======================================================
# # Heatmap
# # ======================================================

# plot_df = df.set_index("Version")[criteria]

# fig, ax = plt.subplots(figsize=(10, max(3, len(plot_df) * 0.6)))

# im = ax.imshow(plot_df.values)

# # Labels
# ax.set_xticks(range(len(criteria)))
# ax.set_xticklabels(criteria, rotation=30, ha="right")

# ax.set_yticks(range(len(plot_df.index)))
# ax.set_yticklabels(plot_df.index)

# # Numbers inside cells
# for i in range(plot_df.shape[0]):
#     for j in range(plot_df.shape[1]):
#         ax.text(
#             j,
#             i,
#             f"{plot_df.iloc[i, j]:.2f}",
#             ha="center",
#             va="center",
#             fontsize=9,
#             color="black",
#         )

# cbar = plt.colorbar(im)
# cbar.set_label("Average Score")

# plt.tight_layout()

# plt.savefig(
#     Figure_Folder / "summary_heatmap.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.close()


# ======================================================
# Save
# ======================================================

# output_csv = BASE_FOLDER / "summary_scores.csv"
# output_excel = BASE_FOLDER / "summary_scores.xlsx"

# df.to_csv(output_csv, index=False)
# df.to_excel(output_excel, index=False)

# print(df)

# print(f"\nSaved to:\n{output_csv}\n{output_excel}")