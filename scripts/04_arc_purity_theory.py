import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# data
df = pd.read_excel("../data/processed/Base maestra.xlsx")
df["Mes"] = df["Mes"].str.strip()

MILLS  = sorted(df["Ingenio"].unique())
ZAFRAS = sorted(df["Zafra"].unique())
MILL_COLORS  = {m: c for m, c in zip(MILLS,  plt.cm.tab10.colors)}
ZAFRA_COLORS = {z: c for z, c in zip(ZAFRAS, plt.cm.tab10.colors)}

os.makedirs("../outputs/figures/Q8_arc_purity", exist_ok=True)


# ── Q8: AR/C vs Pureza Real — validate theory across zafras and mills ──────────

# -- Figure 1: 5 subplots by zafra, all mills colored --
fig1, axes1 = plt.subplots(1, 5, figsize=(22, 6))
fig1.suptitle("AR/C vs Pureza Real — per zafra", fontsize=14, fontweight="bold")

for i, zafra in enumerate(ZAFRAS):
    ax = axes1[i]
    zd = df[df["Zafra"] == zafra]

    x = zd["AR_Cenizas"].values
    y = zd["Pureza_Real"].values

    for mill in MILLS:
        md = zd[zd["Ingenio"] == mill]
        ax.scatter(md["AR_Cenizas"], md["Pureza_Real"],
                   color=MILL_COLORS[mill], label=mill, alpha=0.75, s=50, zorder=3)

    # regression line
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5, linestyle="--", zorder=4)

    r, p = stats.pearsonr(x, y)
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.annotate(f"r = {r:.3f}\n{p_str}\nn = {len(x)}",
                xy=(0.05, 0.90), xycoords="axes fraction",
                fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

    ax.axvline(1.0, color="red", linewidth=1, linestyle=":", alpha=0.7)
    ax.set_title(zafra, fontsize=9, fontweight="bold")
    ax.set_xlabel("AR/C", fontsize=8)
    ax.set_ylabel("Pureza Real (%)", fontsize=8)
    ax.tick_params(labelsize=7)

# single legend
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MILL_COLORS[m],
                      markersize=7, label=m) for m in MILLS]
fig1.legend(handles=handles, loc="lower center", ncol=8, fontsize=7, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=[0, 0.06, 1, 0.96])
fig1.savefig("../outputs/figures/Q8_arc_purity/08a_arc_purity_per_zafra.png", dpi=150, bbox_inches="tight")
print("Figure saved: Q8_arc_purity/08a_arc_purity_per_zafra.png")
plt.close(fig1)


# -- Figure 2: 8 subplots by mill, all zafras colored --
fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
fig2.suptitle("AR/C vs Pureza Real — per mill", fontsize=14, fontweight="bold")
axes2 = axes2.flatten()

for i, mill in enumerate(MILLS):
    ax = axes2[i]
    md = df[df["Ingenio"] == mill]

    x = md["AR_Cenizas"].values
    y = md["Pureza_Real"].values

    for zafra in ZAFRAS:
        zd = md[md["Zafra"] == zafra]
        ax.scatter(zd["AR_Cenizas"], zd["Pureza_Real"],
                   color=ZAFRA_COLORS[zafra], label=zafra, alpha=0.75, s=50, zorder=3)

    # regression line across all zafras for this mill
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5, linestyle="--", zorder=4)

    r, p = stats.pearsonr(x, y)
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.annotate(f"r = {r:.3f}\n{p_str}\nn = {len(x)}",
                xy=(0.05, 0.90), xycoords="axes fraction",
                fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

    ax.axvline(1.0, color="red", linewidth=1, linestyle=":", alpha=0.7)
    ax.set_title(mill, fontsize=10, fontweight="bold")
    ax.set_xlabel("AR/C", fontsize=8)
    ax.set_ylabel("Pureza Real (%)", fontsize=8)
    ax.tick_params(labelsize=8)

# single legend
handles2 = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ZAFRA_COLORS[z],
                       markersize=7, label=z) for z in ZAFRAS]
fig2.legend(handles=handles2, title="Zafra", loc="lower center",
            ncol=5, fontsize=8, title_fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig2.savefig("../outputs/figures/Q8_arc_purity/08b_arc_purity_per_mill.png", dpi=150, bbox_inches="tight")
print("Figure saved: Q8_arc_purity/08b_arc_purity_per_mill.png")
plt.close(fig2)
