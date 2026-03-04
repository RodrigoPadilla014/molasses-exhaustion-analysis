import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats
import os

# data
df = pd.read_excel("../data/processed/Base maestra.xlsx")
df["Mes"] = df["Mes"].str.strip()

MONTH_ORDER = ["Noviembre", "Diciembre", "Enero", "Febrero", "Marzo", "Abril", "Mayo"]
MILLS  = sorted(df["Ingenio"].unique())
ZAFRAS = sorted(df["Zafra"].unique())

ZAFRA_COLORS = {z: c for z, c in zip(ZAFRAS, plt.cm.tab10.colors)}
MILL_COLORS  = {m: c for m, c in zip(MILLS,  plt.cm.tab10.colors)}

os.makedirs("../outputs/figures/Q4_losses", exist_ok=True)

# gap = Pol - Sacarosa per sample
df["gap_pol_sac"] = df["Pol"] - df["Sacarosa"]


# ── Q4.a: systematic error between Pol and Sacarosa HPLC ──────────────────────

# overall mean for reference
overall_mean = df["gap_pol_sac"].mean()
print(f"Overall mean gap (Pol - Sacarosa): {overall_mean:.3f}")

# -- Figure 1: boxplot by mill --
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.suptitle("Pol − Sacarosa HPLC gap by mill", fontsize=13, fontweight="bold")

box_data = [df[df["Ingenio"] == m]["gap_pol_sac"].dropna().values for m in MILLS]

bp = ax1.boxplot(
    box_data,
    labels=MILLS,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
)

for patch, mill in zip(bp["boxes"], MILLS):
    patch.set_facecolor(MILL_COLORS[mill])
    patch.set_alpha(0.75)

ax1.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Zero (no gap)")
ax1.axhline(overall_mean, color="red", linestyle=":", linewidth=1.2, alpha=0.8, label=f"Overall mean ({overall_mean:.2f})")
ax1.set_ylabel("Pol − Sacarosa", fontsize=10)
ax1.tick_params(axis="x", labelrotation=30, labelsize=8)
ax1.legend(fontsize=8)

plt.tight_layout()
fig1.savefig("../outputs/figures/Q4_losses/04a_gap_boxplot.png", dpi=150, bbox_inches="tight")
print("Figure saved: Q4_losses/04a_gap_boxplot.png")
plt.close(fig1)


# -- Figure 2: monthly trajectories per mill (8 subplots, one line per zafra) --
fig2, axes = plt.subplots(2, 4, figsize=(20, 10))
fig2.suptitle("Pol − Sacarosa HPLC gap — monthly trajectory per mill", fontsize=14, fontweight="bold")
axes = axes.flatten()

for i, mill in enumerate(MILLS):
    ax = axes[i]
    mill_data = df[df["Ingenio"] == mill].copy()

    for zafra in ZAFRAS:
        zd = mill_data[mill_data["Zafra"] == zafra].copy()
        zd["Mes"] = pd.Categorical(zd["Mes"], categories=MONTH_ORDER, ordered=True)
        zd = zd.dropna(subset=["Mes"]).sort_values("Mes")

        if zd.empty:
            continue

        ax.plot(
            zd["Mes"],
            zd["gap_pol_sac"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=ZAFRA_COLORS[zafra],
            label=zafra,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title(mill, fontsize=10, fontweight="bold")
    ax.set_ylabel("Pol − Sacarosa", fontsize=8)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

handles = [
    plt.Line2D([0], [0], color=ZAFRA_COLORS[z], marker="o", markersize=4, linewidth=1.5, label=z)
    for z in ZAFRAS
]
fig2.legend(handles=handles, title="Zafra", loc="lower center",
            ncol=5, fontsize=8, title_fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig2.savefig("../outputs/figures/Q4_losses/04b_gap_monthly.png", dpi=150, bbox_inches="tight")
print("Figure saved: Q4_losses/04b_gap_monthly.png")
plt.close(fig2)


# ── Q4.b: what drives the Pol drift? ──────────────────────────────────────────
# Perd_Indet = PerdSac_MF - PerdPol_MF
# Since HPLC is the ground truth, we ask: what composition variables explain
# how much Pol deviates from HPLC?

df["Perd_Indet_calc"] = df["PerdSac_MF"] - df["PerdPol_MF"]
df_q4b = df.dropna(subset=["Perd_Indet_calc"]).copy()

# merge cluster labels
labels = pd.read_csv("../outputs/cluster_labels.csv")
labels["Mes"] = labels["Mes"].str.strip()
df_q4b = df_q4b.merge(labels, on=["Ingenio", "Zafra", "Mes"], how="left")

# mielera flag (old definition)
df_q4b["mielera"] = df_q4b["kg/t_MF"].apply(
    lambda x: "Mielera" if pd.notna(x) and x > 40 else "Non-mielera"
)

# variables that could explain Pol drift
DRIVERS = {
    "AR_Fehling":       "AR Fehling",
    "Fructosa/Glucosa": "Fructosa/Glucosa",
    "No_Pol":           "No-Pol",
    "pH":               "pH",
}

CLUSTER_LABELS = {0: "Cluster 1\nMielera", 1: "Cluster 2\nNormal", 2: "Cluster 3\nDifficult"}
CLUSTER_COLORS = {0: "#4CAF50", 1: "#FFC107", 2: "#F44336"}


def scatter_q4b(data, x_col, x_label, groupby, group_vals, group_colors, group_labels, fname, title):
    n_groups = len(group_vals)
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 6))
    if n_groups == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for i, gval in enumerate(group_vals):
        ax = axes[i]
        gd = data[data[groupby] == gval][[x_col, "Perd_Indet_calc"]].dropna()
        x = gd[x_col].values
        y = gd["Perd_Indet_calc"].values

        ax.scatter(x, y, color=group_colors[gval], alpha=0.75, s=50, zorder=3)

        if len(x) >= 5:
            r, p = stats.pearsonr(x, y)
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5, linestyle="--")
            p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            ax.annotate(f"r = {r:.3f}\n{p_str}\nn = {len(x)}",
                        xy=(0.05, 0.90), xycoords="axes fraction", fontsize=8, verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))
        else:
            ax.annotate(f"n = {len(x)}\ninsufficient data",
                        xy=(0.05, 0.90), xycoords="axes fraction", fontsize=8, verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

        ax.set_title(group_labels[gval], fontsize=9, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel("Perd_Indet (PerdSac - PerdPol)", fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(f"../outputs/figures/Q4_losses/{fname}.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved: Q4_losses/{fname}.png")
    plt.close(fig)


for col, col_label in DRIVERS.items():

    # safe filename — remove special characters
    col_safe = col.replace("/", "_").replace(" ", "_")

    # -- Lens 1: per zafra (5 subplots) --
    zafra_colors = {z: c for z, c in zip(ZAFRAS, plt.cm.tab10.colors)}
    zafra_labels = {z: z for z in ZAFRAS}
    scatter_q4b(df_q4b, col, col_label, "Zafra", ZAFRAS,
                zafra_colors, zafra_labels,
                f"04c_{col_safe}_per_zafra", f"Perd_Indet vs {col_label} — per zafra")

    # -- Lens 2: per mill (8 subplots) --
    mill_colors = MILL_COLORS
    mill_labels = {m: m for m in MILLS}
    scatter_q4b(df_q4b, col, col_label, "Ingenio", MILLS,
                mill_colors, mill_labels,
                f"04d_{col_safe}_per_mill", f"Perd_Indet vs {col_label} — per mill")

    # -- Lens 3: per mielera group (2 subplots) --
    mielera_colors = {"Mielera": "#f7a1a1", "Non-mielera": "#a8d5e2"}
    mielera_labels = {"Mielera": "Mielera (kg/t > 40)", "Non-mielera": "Non-mielera (kg/t <= 40)"}
    scatter_q4b(df_q4b, col, col_label, "mielera", ["Mielera", "Non-mielera"],
                mielera_colors, mielera_labels,
                f"04e_{col_safe}_per_mielera", f"Perd_Indet vs {col_label} — mielera split")

    # -- Lens 4: per cluster (3 subplots) --
    df_q4b_cl = df_q4b[df_q4b["cluster"].notna()].copy()
    df_q4b_cl["cluster"] = df_q4b_cl["cluster"].astype(int)
    clusters = sorted(df_q4b_cl["cluster"].unique())
    scatter_q4b(df_q4b_cl, col, col_label, "cluster", clusters,
                CLUSTER_COLORS, CLUSTER_LABELS,
                f"04f_{col_safe}_per_cluster", f"Perd_Indet vs {col_label} — per cluster")
