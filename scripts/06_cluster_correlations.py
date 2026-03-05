import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# data
df = pd.read_excel("../data/processed/Base maestra.xlsx")
df["Mes"] = df["Mes"].str.strip()
df["DPO"] = df["Pureza_Real"] - df["Pureza_Obj"]

# merge cluster labels
labels = pd.read_csv("../outputs/cluster_labels.csv")
labels["Mes"] = labels["Mes"].str.strip()
df = df.merge(labels, on=["Ingenio", "Zafra", "Mes"], how="left")

# drop rows with no cluster (1 sample dropped during clustering due to null)
df = df[df["cluster"].notna()].copy()
df["cluster"] = df["cluster"].astype(int)

MILLS    = sorted(df["Ingenio"].unique())
CLUSTERS = sorted(df["cluster"].unique())
MILL_COLORS    = {m: c for m, c in zip(MILLS,    plt.cm.tab10.colors)}
CLUSTER_COLORS = {0: "#F44336", 1: "#4CAF50", 2: "#FF9800"}
CLUSTER_LABELS = {0: "Cluster 1 — Mielera", 1: "Cluster 2 — Normal", 2: "Cluster 3 — No Mielera"}

os.makedirs("../outputs/figures/Q9_clustering", exist_ok=True)

# drop rows with no viscosity for Q6
df_visc = df.dropna(subset=["Viscosidad_40C"]).copy()


def scatter_per_cluster(data, x_col, y_col, x_label, y_label, fname, title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, cluster in enumerate(CLUSTERS):
        ax = axes[i]
        cd = data[data["cluster"] == cluster]
        x = cd[x_col].dropna().values
        y = cd[y_col].dropna().values

        # align x and y on same rows
        pair = data[data["cluster"] == cluster][[x_col, y_col]].dropna()
        x = pair[x_col].values
        y = pair[y_col].values

        ax.scatter(x, y, color=CLUSTER_COLORS[cluster], alpha=0.75, s=55, zorder=3)

        if len(x) >= 5:
            r, p = stats.pearsonr(x, y)
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5, linestyle="--")
            p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            ax.annotate(f"r = {r:.3f}\n{p_str}\nn = {len(x)}",
                        xy=(0.05, 0.90), xycoords="axes fraction", fontsize=9, verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))
        else:
            ax.annotate(f"n = {len(x)}\ninsufficient data",
                        xy=(0.05, 0.90), xycoords="axes fraction", fontsize=9, verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

        ax.set_title(CLUSTER_LABELS[cluster], fontsize=10, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(f"../outputs/figures/Q9_clustering/{fname}.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved: Q9_clustering/{fname}.png")
    plt.close(fig)


# ── Q6: Viscosity 40C vs DPO — per cluster ────────────────────────────────────
scatter_per_cluster(
    df_visc, "Viscosidad_40C", "DPO",
    "Viscosity at 40C (Pa.s)", "DPO",
    "09f_visc_dpo_per_cluster",
    "Viscosity 40C vs DPO — per cluster"
)


# ── Q7: AR/C vs Color — per cluster ───────────────────────────────────────────
scatter_per_cluster(
    df, "AR_Cenizas", "Color",
    "AR/C", "Color (IU)",
    "09g_arc_color_per_cluster",
    "AR/C vs Color — per cluster"
)


# ── Q8: AR/C vs Pureza Real — per cluster ─────────────────────────────────────
scatter_per_cluster(
    df, "AR_Cenizas", "Pureza_Real",
    "AR/C", "Pureza Real (%)",
    "09h_arc_purity_per_cluster",
    "AR/C vs Pureza Real — per cluster"
)


# ── CLUSTER TRAJECTORY PER MILL ACROSS ZAFRAS ─────────────────────────────────
# For each mill-zafra, assign the most frequent cluster (mode) of that season.
# Then plot how each mill moves between clusters over the 5 zafras.

ZAFRAS = sorted(df["Zafra"].unique())

trajectory = (
    df.groupby(["Ingenio", "Zafra"])["cluster"]
    .agg(lambda x: x.mode()[0])
    .reset_index()
)
trajectory["cluster_label"] = trajectory["cluster"] + 1

# -- Figure: heatmap — mills on Y, zafras on X, color = cluster --
pivot = trajectory.pivot(index="Ingenio", columns="Zafra", values="cluster_label")

# custom colormap: 1=green (mielera), 2=yellow (normal), 3=red (difficult)
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#F44336", "#4CAF50", "#FF9800"])

fig_traj, ax_traj = plt.subplots(figsize=(10, 6))
fig_traj.suptitle("Cluster trajectory per mill across zafras", fontsize=13, fontweight="bold")

im = ax_traj.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0.5, vmax=3.5)

ax_traj.set_xticks(range(len(pivot.columns)))
ax_traj.set_xticklabels(pivot.columns, fontsize=9, rotation=30)
ax_traj.set_yticks(range(len(pivot.index)))
ax_traj.set_yticklabels(pivot.index, fontsize=9)

# annotate each cell with cluster number
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.iloc[i, j]
        ax_traj.text(j, i, str(int(val)), ha="center", va="center", fontsize=11, fontweight="bold", color="white")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#F44336", label="Cluster 1 — Mielera"),
    Patch(facecolor="#4CAF50", label="Cluster 2 — Normal"),
    Patch(facecolor="#FF9800", label="Cluster 3 — No Mielera"),
]
ax_traj.legend(handles=legend_elements, loc="upper right", fontsize=8, bbox_to_anchor=(1.35, 1))

plt.tight_layout()
fig_traj.savefig("../outputs/figures/Q9_clustering/09i_cluster_trajectory.png", dpi=150, bbox_inches="tight")
print("Figure saved: Q9_clustering/09i_cluster_trajectory.png")
plt.close(fig_traj)

# print trajectory table
print("\n-- Cluster trajectory per mill (mode per zafra) --")
pivot = trajectory.pivot(index="Ingenio", columns="Zafra", values="cluster_label")
print(pivot.to_string())
