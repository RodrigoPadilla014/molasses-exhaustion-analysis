import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
