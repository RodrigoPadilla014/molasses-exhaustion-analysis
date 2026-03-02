import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ── DATA ──────────────────────────────────────────────────────────────────────

df = pd.read_excel("../data/processed/Base maestra.xlsx")

# Clean up whitespace in month names (e.g. "Mayo " -> "Mayo")
df["Mes"] = df["Mes"].str.strip()

# Harvest months in calendar order (Nov = start, May = end)
MONTH_ORDER = ["Noviembre", "Diciembre", "Enero", "Febrero", "Marzo", "Abril", "Mayo"]

# Mills sorted alphabetically for consistent colors across plots
MILLS = sorted(df["Ingenio"].unique())
ZAFRAS = sorted(df["Zafra"].unique())

# One color per zafra
ZAFRA_COLORS = {z: c for z, c in zip(ZAFRAS, plt.cm.tab10.colors)}

# One color per mill
MILL_COLORS = {m: c for m, c in zip(MILLS, plt.cm.tab10.colors)}


# ── QUESTION 1: AR/C STABILITY PER MILL ───────────────────────────────────────
# Are mills consistently above AR/C = 1.0? Are they improving or deteriorating?

os.makedirs("../outputs/figures/Q1_ARC",          exist_ok=True)
os.makedirs("../outputs/figures/Q2_DPO",          exist_ok=True)
os.makedirs("../outputs/figures/Q3_correlations", exist_ok=True)

# Mean AR/C per mill per zafra (average across all months of that harvest)
# Used in both figures below
annual = df.groupby(["Ingenio", "Zafra"])["AR_Cenizas"].mean().reset_index()
annual.columns = ["Ingenio", "Zafra", "AR_C_mean"]

# ── FIGURE 1: monthly — 8 subplots, one per mill ──────────────────────────────
# Each subplot shows 5 lines (one per zafra) across the harvest months.
# Lets you see within-harvest trajectory and how it shifted year over year.

fig1, axes = plt.subplots(2, 4, figsize=(20, 10))
fig1.suptitle("AR/C Monthly Trajectory per Mill", fontsize=14, fontweight="bold")
axes = axes.flatten()

for i, mill in enumerate(MILLS):
    ax = axes[i]
    mill_data = df[df["Ingenio"] == mill].copy()

    for zafra in ZAFRAS:
        zafra_data = mill_data[mill_data["Zafra"] == zafra].copy()

        # Sort months in harvest order, drop any month not in the list
        zafra_data["Mes"] = pd.Categorical(zafra_data["Mes"], categories=MONTH_ORDER, ordered=True)
        zafra_data = zafra_data.dropna(subset=["Mes"]).sort_values("Mes")

        if zafra_data.empty:
            continue

        ax.plot(
            zafra_data["Mes"],
            zafra_data["AR_Cenizas"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=ZAFRA_COLORS[zafra],
            label=zafra,
        )

    # Reference line at 1.0 — below this crystallization becomes harder
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_title(mill, fontsize=10, fontweight="bold")
    ax.set_ylabel("AR/C", fontsize=8)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

# Single legend for all zafra colors
handles = [
    plt.Line2D([0], [0], color=ZAFRA_COLORS[z], marker="o", markersize=4, linewidth=1.5, label=z)
    for z in ZAFRAS
]
fig1.legend(handles=handles, title="Zafra", loc="lower center",
            ncol=5, fontsize=8, title_fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig1.savefig("../outputs/figures/Q1_ARC/01a_ARC_monthly.png", dpi=150, bbox_inches="tight")
print("Figure saved: outputs/figures/Q1_ARC/01a_ARC_monthly.png")
plt.close(fig1)

# ── FIGURE 2: annual — one line per mill across the 5 zafras ──────────────────
# Shows multi-year trend: which mills are improving, deteriorating, or flat.

fig2, ax_annual = plt.subplots(figsize=(10, 6))
fig2.suptitle("AR/C Annual Mean per Mill", fontsize=14, fontweight="bold")

for mill in MILLS:
    mill_annual = annual[annual["Ingenio"] == mill].sort_values("Zafra")
    ax_annual.plot(
        mill_annual["Zafra"],
        mill_annual["AR_C_mean"],
        marker="o",
        markersize=6,
        linewidth=2,
        color=MILL_COLORS[mill],
        label=mill,
    )

# Reference line at 1.0
ax_annual.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6, label="Ref. 1.0")
ax_annual.set_xlabel("Zafra", fontsize=9)
ax_annual.set_ylabel("Mean AR/C", fontsize=9)
ax_annual.tick_params(axis="x", labelrotation=30, labelsize=8)
ax_annual.legend(loc="upper left", fontsize=8, ncol=2)
ax_annual.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

plt.tight_layout()
fig2.savefig("../outputs/figures/Q1_ARC/01b_ARC_annual.png", dpi=150, bbox_inches="tight")
print("Figure saved: outputs/figures/Q1_ARC/01b_ARC_annual.png")
plt.close(fig2)

# ── DESCRIPTIVE STATS (annual level) ──────────────────────────────────────────
# For each mill: mean, std, min, max across the 5 zafra averages,
# and how many zafras they stayed above AR/C = 1.0

stats = annual.groupby("Ingenio")["AR_C_mean"].agg(
    mean="mean",
    std="std",
    min="min",
    max="max",
).round(3)

stats["zafras_above_1"] = (
    annual[annual["AR_C_mean"] > 1.0]
    .groupby("Ingenio")
    .size()
    .reindex(stats.index, fill_value=0)
)

stats[f"zafras_above_1"] = stats["zafras_above_1"].astype(str) + f"/{len(ZAFRAS)}"

print("\n-- AR/C Annual Descriptive Statistics per Mill --")
print(stats.to_string())


# ── QUESTION 2: DPO EVOLUTION ─────────────────────────────────────────────────
# Has the gap between real purity and objective purity narrowed, held, or worsened?
# Which mills closed it the fastest?

# DPO = Pureza_Real - Pureza_Obj (both already in the dataset)
df["DPO"] = df["Pureza_Real"] - df["Pureza_Obj"]

# Annual mean DPO per mill per zafra
annual_dpo = df.groupby(["Ingenio", "Zafra"])["DPO"].mean().reset_index()
annual_dpo.columns = ["Ingenio", "Zafra", "DPO_mean"]

# ── FIGURE 3: monthly DPO — 8 subplots, one per mill ──────────────────────────
# Each subplot shows 5 lines (one per zafra) across harvest months.
# Reference bands show what counts as normal (5-7) and top performance (2-3).

fig3, axes3 = plt.subplots(2, 4, figsize=(20, 10))
fig3.suptitle("DPO Monthly Trajectory per Mill", fontsize=14, fontweight="bold")
axes3 = axes3.flatten()

for i, mill in enumerate(MILLS):
    ax = axes3[i]
    mill_data = df[df["Ingenio"] == mill].copy()

    for zafra in ZAFRAS:
        zafra_data = mill_data[mill_data["Zafra"] == zafra].copy()

        zafra_data["Mes"] = pd.Categorical(zafra_data["Mes"], categories=MONTH_ORDER, ordered=True)
        zafra_data = zafra_data.dropna(subset=["Mes"]).sort_values("Mes")

        if zafra_data.empty:
            continue

        ax.plot(
            zafra_data["Mes"],
            zafra_data["DPO"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=ZAFRA_COLORS[zafra],
            label=zafra,
        )

    # Shaded band for "normal" DPO range (5-7) from the literature
    ax.axhspan(5, 7, color="orange", alpha=0.12, label="Normal (5-7)")
    # Shaded band for top-performer range (2-3)
    ax.axhspan(2, 3, color="green", alpha=0.12, label="Top (2-3)")

    ax.set_title(mill, fontsize=10, fontweight="bold")
    ax.set_ylabel("DPO", fontsize=8)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

# Single legend for zafra colors
handles3 = [
    plt.Line2D([0], [0], color=ZAFRA_COLORS[z], marker="o", markersize=4, linewidth=1.5, label=z)
    for z in ZAFRAS
]
# Add reference band patches to legend
from matplotlib.patches import Patch
handles3 += [
    Patch(facecolor="orange", alpha=0.3, label="Normal (5-7)"),
    Patch(facecolor="green",  alpha=0.3, label="Top (2-3)"),
]
fig3.legend(handles=handles3, title="Zafra", loc="lower center",
            ncol=7, fontsize=8, title_fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig3.savefig("../outputs/figures/Q2_DPO/02a_DPO_monthly.png", dpi=150, bbox_inches="tight")
print("\nFigure saved: outputs/figures/Q2_DPO/02a_DPO_monthly.png")
plt.close(fig3)

# ── FIGURE 4: annual mean DPO per mill across 5 zafras ────────────────────────
# Shows whether each mill is trending toward better exhaustion over the years.

fig4, ax_dpo = plt.subplots(figsize=(10, 6))
fig4.suptitle("DPO Annual Mean per Mill", fontsize=14, fontweight="bold")

for mill in MILLS:
    mill_dpo = annual_dpo[annual_dpo["Ingenio"] == mill].sort_values("Zafra")
    ax_dpo.plot(
        mill_dpo["Zafra"],
        mill_dpo["DPO_mean"],
        marker="o",
        markersize=6,
        linewidth=2,
        color=MILL_COLORS[mill],
        label=mill,
    )

# Reference bands
ax_dpo.axhspan(5, 7, color="orange", alpha=0.12, label="Normal (5-7)")
ax_dpo.axhspan(2, 3, color="green",  alpha=0.12, label="Top (2-3)")
ax_dpo.set_xlabel("Zafra", fontsize=9)
ax_dpo.set_ylabel("Mean DPO", fontsize=9)
ax_dpo.tick_params(axis="x", labelrotation=30, labelsize=8)
ax_dpo.legend(loc="upper right", fontsize=8, ncol=2)
ax_dpo.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

plt.tight_layout()
fig4.savefig("../outputs/figures/Q2_DPO/02b_DPO_annual.png", dpi=150, bbox_inches="tight")
print("Figure saved: outputs/figures/Q2_DPO/02b_DPO_annual.png")
plt.close(fig4)

# ── DESCRIPTIVE STATS: DPO (annual level) ─────────────────────────────────────
# Mean, std, min, max per mill across the 5 zafras.
# Delta = first zafra mean minus last zafra mean — positive means improvement.

dpo_stats = annual_dpo.groupby("Ingenio")["DPO_mean"].agg(
    mean="mean",
    std="std",
    min="min",
    max="max",
).round(3)

first_zafra = ZAFRAS[0]
last_zafra  = ZAFRAS[-1]

first_dpo = annual_dpo[annual_dpo["Zafra"] == first_zafra].set_index("Ingenio")["DPO_mean"]
last_dpo  = annual_dpo[annual_dpo["Zafra"] == last_zafra].set_index("Ingenio")["DPO_mean"]

# Positive delta = DPO went down = mill improved exhaustion
dpo_stats["delta_first_to_last"] = (first_dpo - last_dpo).round(3)

print("\n-- DPO Annual Descriptive Statistics per Mill --")
print(f"   (delta = {first_zafra} mean minus {last_zafra} mean; positive = improvement)")
print(dpo_stats.to_string())


# ── QUESTION 3: SEASONALITY, ZAFRA MIELERA, AND CORRELATIONS ─────────────────

from scipy import stats
import numpy as np

# Pol%cana column has encoding issues in its name on Windows — locate it by position
POL_CANA = df.columns[31]

os.makedirs("../outputs/figures/Q3_seasonality",   exist_ok=True)
os.makedirs("../outputs/figures/Q3_mielera",        exist_ok=True)
os.makedirs("../outputs/figures/Q3_correlations",   exist_ok=True)

# ── HARVEST PHASE DEFINITION ──────────────────────────────────────────────────
# Early: Nov-Dec | Mid: Jan-Feb-Mar | Late: Apr-May

phase_map = {
    "Noviembre": "Early",
    "Diciembre": "Early",
    "Enero":     "Mid",
    "Febrero":   "Mid",
    "Marzo":     "Mid",
    "Abril":     "Late",
    "Mayo":      "Late",
}
PHASE_ORDER = ["Early", "Mid", "Late"]
df["Phase"] = df["Mes"].map(phase_map)

# ── SECTION 1: SEASONALITY ────────────────────────────────────────────────────
# Do AR/C and Pureza Real shift systematically as the harvest progresses?

for var, ylabel, fname in [
    ("AR_Cenizas",  "AR/C",            "03a_ARC_by_phase"),
    ("Pureza_Real", "Pureza Real (%)",  "03b_purity_by_phase"),
]:
    phase_data = [df[df["Phase"] == p][var].dropna().values for p in PHASE_ORDER]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"{ylabel} by Harvest Phase", fontsize=13, fontweight="bold")

    bp = ax.boxplot(
        phase_data,
        labels=PHASE_ORDER,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )

    for patch, color in zip(bp["boxes"], ["#a8d5e2", "#f7c59f", "#e8a0bf"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xlabel("Harvest Phase", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    if var == "AR_Cenizas":
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6, label="Ref. 1.0")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(f"../outputs/figures/Q3_seasonality/{fname}.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved: outputs/figures/Q3_seasonality/{fname}.png")
    plt.close(fig)

# ── SECTION 2: ZAFRA MIELERA ──────────────────────────────────────────────────
# Label each month as mielera (kg/t MF > 40) or non-mielera and compare groups

MIELERA_THRESHOLD = 40

df["mielera"] = df["kg/t_MF"].apply(
    lambda x: "Mielera (>40)" if pd.notna(x) and x > MIELERA_THRESHOLD else "Non-mielera (<=40)"
)

# Drop the 1 row with null kg/t_MF before group comparisons
df_mielera = df[df["kg/t_MF"].notna()].copy()

n_mielera     = (df_mielera["mielera"] == "Mielera (>40)").sum()
n_non_mielera = (df_mielera["mielera"] == "Non-mielera (<=40)").sum()
print(f"\nMielera months: {n_mielera} | Non-mielera months: {n_non_mielera}")

for var, ylabel, fname in [
    ("AR_Cenizas",  "AR/C",            "03c_ARC_mielera"),
    ("Pureza_Real", "Pureza Real (%)",  "03d_purity_mielera"),
]:
    groups     = ["Non-mielera (<=40)", "Mielera (>40)"]
    group_data = [df_mielera[df_mielera["mielera"] == g][var].dropna().values for g in groups]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"{ylabel} — Mielera vs. Non-mielera", fontsize=13, fontweight="bold")

    bp = ax.boxplot(
        group_data,
        labels=groups,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )

    for patch, color in zip(bp["boxes"], ["#a8d5e2", "#f7a1a1"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_ylabel(ylabel, fontsize=10)

    if var == "AR_Cenizas":
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6, label="Ref. 1.0")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(f"../outputs/figures/Q3_mielera/{fname}.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved: outputs/figures/Q3_mielera/{fname}.png")
    plt.close(fig)

    # T-test: is the mean difference between mielera and non-mielera significant?
    t_stat, p_val = stats.ttest_ind(group_data[0], group_data[1], equal_var=False)
    print(f"\n-- T-test: {ylabel} Mielera vs. Non-mielera --")
    print(f"   Non-mielera mean: {group_data[0].mean():.3f}  (n={len(group_data[0])})")
    print(f"   Mielera mean:     {group_data[1].mean():.3f}  (n={len(group_data[1])})")
    print(f"   t = {t_stat:.3f}  p = {p_val:.4f}")

# ── SECTION 3: CORRELATIONS ───────────────────────────────────────────────────
# Pooled scatter + within-mill breakdown for all 5 pairs.
# For Pol%cana pairs: additionally split by mielera/non-mielera.

corr_pairs = [
    ("AR_Cenizas", "Pureza_Real", "AR/C",            "Pureza Real (%)", "03e_purity_vs_ARC"),
    ("kg/t_MF",    "Pureza_Real", "kg/t MF",          "Pureza Real (%)", "03f_purity_vs_kgtMF"),
    ("kg/t_MF",    "AR_Cenizas",  "kg/t MF",          "AR/C",            "03g_ARC_vs_kgtMF"),
    (POL_CANA,     "Pureza_Real", "Pol%cana (kg/t)",   "Pureza Real (%)", "03h_purity_vs_polcana"),
    (POL_CANA,     "AR_Cenizas",  "Pol%cana (kg/t)",   "AR/C",            "03i_ARC_vs_polcana"),
]

pooled_results = []

for x_col, y_col, x_label, y_label, fig_name in corr_pairs:

    pair_data = df[[x_col, y_col, "Ingenio", "mielera"]].dropna(subset=[x_col, y_col]).copy()
    x = pair_data[x_col].values
    y = pair_data[y_col].values

    # Pooled Pearson r
    r, p = stats.pearsonr(x, y)

    # Regression line
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"{y_label} vs. {x_label}", fontsize=13, fontweight="bold")

    for mill in MILLS:
        mill_pts = pair_data[pair_data["Ingenio"] == mill]
        ax.scatter(
            mill_pts[x_col],
            mill_pts[y_col],
            color=MILL_COLORS[mill],
            label=mill,
            alpha=0.75,
            s=50,
            zorder=3,
        )

    ax.plot(x_line, y_line, color="black", linewidth=1.5, linestyle="--", zorder=4)

    # Zafra mielera threshold line on kg/t MF plots
    if x_col == "kg/t_MF":
        ax.axvline(40, color="red", linewidth=1.2, linestyle=":", alpha=0.8, label="Zafra mielera (40 kg/t)")

    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.annotate(
        f"r = {r:.3f}\n{p_str}\nn = {len(x)}",
        xy=(0.05, 0.90),
        xycoords="axes fraction",
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    plt.tight_layout()
    fig.savefig(f"../outputs/figures/Q3_correlations/{fig_name}.png", dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: outputs/figures/Q3_correlations/{fig_name}.png")
    plt.close(fig)

    pooled_results.append({
        "pair": f"{y_label} vs. {x_label}",
        "r":    round(r, 3),
        "p":    round(p, 4),
        "n":    len(x),
    })

    # Within-mill correlations
    print(f"   Within-mill r ({y_label} vs. {x_label}):")
    for mill in MILLS:
        mill_data = pair_data[pair_data["Ingenio"] == mill]
        if len(mill_data) < 4:
            print(f"      {mill}: insufficient data")
            continue
        r_m, p_m = stats.pearsonr(mill_data[x_col].values, mill_data[y_col].values)
        p_str_m = "p < 0.001" if p_m < 0.001 else f"p = {p_m:.3f}"
        print(f"      {mill}: r = {r_m:.3f}  {p_str_m}  (n={len(mill_data)})")

    # For Pol%cana pairs: split by mielera/non-mielera
    if x_col == POL_CANA:
        print(f"   Pol%cana split by zafra mielera ({y_label}):")
        for group in ["Non-mielera (<=40)", "Mielera (>40)"]:
            g_data = pair_data[pair_data["mielera"] == group]
            if len(g_data) < 4:
                print(f"      {group}: insufficient data")
                continue
            r_g, p_g = stats.pearsonr(g_data[x_col].values, g_data[y_col].values)
            p_str_g = "p < 0.001" if p_g < 0.001 else f"p = {p_g:.3f}"
            print(f"      {group}: r = {r_g:.3f}  {p_str_g}  (n={len(g_data)})")

# Pooled summary
print("\n-- Pooled Correlation Summary --")
print(pd.DataFrame(pooled_results).to_string(index=False))
