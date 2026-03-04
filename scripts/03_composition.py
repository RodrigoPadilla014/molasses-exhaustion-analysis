import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats
import os

# data
df = pd.read_excel("../data/processed/Base maestra.xlsx")
df["Mes"] = df["Mes"].str.strip()
df["DPO"] = df["Pureza_Real"] - df["Pureza_Obj"]

MILLS  = sorted(df["Ingenio"].unique())
ZAFRAS = sorted(df["Zafra"].unique())
MILL_COLORS = {m: c for m, c in zip(MILLS, plt.cm.tab10.colors)}

os.makedirs("../outputs/figures/Q6_viscosity", exist_ok=True)

# drop rows with no viscosity (2020-2021 season, viscometer not available)
df_visc = df.dropna(subset=["Viscosidad_40C", "DPO"]).copy()
print(f"Rows used for viscosity analysis: {len(df_visc)} (dropped {len(df) - len(df_visc)} nulls)")


# ── Q6: Viscosity at 40°C as a predictor of exhaustion (DPO) ──────────────────

# -- Figures 1-8: one scatter per mill --
for mill in MILLS:
    md = df_visc[df_visc["Ingenio"] == mill].copy()

    x = md["Viscosidad_40C"].values
    y = md["DPO"].values

    r, p = stats.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"Viscosity 40°C vs DPO — {mill}", fontsize=13, fontweight="bold")

    ax.scatter(x, y, color=MILL_COLORS[mill], alpha=0.75, s=55, zorder=3)
    ax.plot(x_line, y_line, color="black", linewidth=1.5, linestyle="--", zorder=4)

    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.annotate(
        f"r = {r:.3f}\n{p_str}\nn = {len(x)}",
        xy=(0.05, 0.90), xycoords="axes fraction",
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("Viscosity at 40°C (Pa·s)", fontsize=10)
    ax.set_ylabel("DPO", fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    fname = mill.replace(" ", "_").replace("ó", "o").replace("ú", "u")
    plt.tight_layout()
    fig.savefig(f"../outputs/figures/Q6_viscosity/06_{fname}_visc_dpo.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved: Q6_viscosity/06_{fname}_visc_dpo.png")
    plt.close(fig)


# -- Figure 9: pooled scatter, all mills colored --
x_all = df_visc["Viscosidad_40C"].values
y_all = df_visc["DPO"].values

r_all, p_all = stats.pearsonr(x_all, y_all)
slope_all, intercept_all = np.polyfit(x_all, y_all, 1)
x_line_all = np.linspace(x_all.min(), x_all.max(), 100)
y_line_all = slope_all * x_line_all + intercept_all

fig9, ax9 = plt.subplots(figsize=(9, 7))
fig9.suptitle("Viscosity 40°C vs DPO — all mills", fontsize=13, fontweight="bold")

for mill in MILLS:
    md = df_visc[df_visc["Ingenio"] == mill]
    ax9.scatter(
        md["Viscosidad_40C"], md["DPO"],
        color=MILL_COLORS[mill], label=mill,
        alpha=0.75, s=55, zorder=3,
    )

ax9.plot(x_line_all, y_line_all, color="black", linewidth=1.5, linestyle="--", zorder=4)

p_str_all = "p < 0.001" if p_all < 0.001 else f"p = {p_all:.3f}"
ax9.annotate(
    f"r = {r_all:.3f}\n{p_str_all}\nn = {len(x_all)}",
    xy=(0.05, 0.90), xycoords="axes fraction",
    fontsize=10, verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
)

ax9.set_xlabel("Viscosity at 40°C (Pa·s)", fontsize=10)
ax9.set_ylabel("DPO", fontsize=10)
ax9.legend(loc="upper right", fontsize=8, ncol=2)
ax9.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

plt.tight_layout()
fig9.savefig("../outputs/figures/Q6_viscosity/06_pooled_visc_dpo.png", dpi=150, bbox_inches="tight")
print("Figure saved: Q6_viscosity/06_pooled_visc_dpo.png")
plt.close(fig9)


# ── Q7: AR/C < 1 and glucose destruction — correlation with Color ──────────────

os.makedirs("../outputs/figures/Q7_color", exist_ok=True)

# flag samples by AR/C threshold
df["arc_group"] = df["AR_Cenizas"].apply(lambda x: "AR/C < 1" if x < 1 else "AR/C >= 1")

# -- Figure 1: boxplot of Color by AR/C group --
groups     = ["AR/C < 1", "AR/C >= 1"]
group_data = [df[df["arc_group"] == g]["Color"].values for g in groups]

n0 = len(group_data[0])
n1 = len(group_data[1])

t_stat, p_val = stats.ttest_ind(group_data[0], group_data[1], equal_var=False)
p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"

fig_box, ax_box = plt.subplots(figsize=(7, 6))
fig_box.suptitle("Color by AR/C group", fontsize=13, fontweight="bold")

bp = ax_box.boxplot(
    group_data,
    tick_labels=[f"AR/C < 1\n(n={n0})", f"AR/C >= 1\n(n={n1})"],
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
)

for patch, color in zip(bp["boxes"], ["#f7a1a1", "#a8d5e2"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax_box.set_ylabel("Color (IU)", fontsize=10)
ax_box.annotate(
    f"t-test: {p_str}",
    xy=(0.5, 0.93), xycoords="axes fraction",
    fontsize=9, ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
)

plt.tight_layout()
fig_box.savefig("../outputs/figures/Q7_color/07a_color_by_arc_group.png", dpi=150, bbox_inches="tight")
print("\nFigure saved: Q7_color/07a_color_by_arc_group.png")
plt.close(fig_box)

print(f"  AR/C < 1 mean Color: {group_data[0].mean():.1f}  (n={n0})")
print(f"  AR/C >= 1 mean Color: {group_data[1].mean():.1f}  (n={n1})")
print(f"  t = {t_stat:.3f}  {p_str}")


# -- Figure 2: scatter AR/C vs Color, colored by mill --
x_arc   = df["AR_Cenizas"].values
y_color = df["Color"].values

r_arc, p_arc = stats.pearsonr(x_arc, y_color)
slope_arc, intercept_arc = np.polyfit(x_arc, y_color, 1)
x_line_arc = np.linspace(x_arc.min(), x_arc.max(), 100)
y_line_arc = slope_arc * x_line_arc + intercept_arc

fig_sc, ax_sc = plt.subplots(figsize=(9, 7))
fig_sc.suptitle("AR/C vs Color — all mills", fontsize=13, fontweight="bold")

for mill in MILLS:
    md = df[df["Ingenio"] == mill]
    ax_sc.scatter(
        md["AR_Cenizas"], md["Color"],
        color=MILL_COLORS[mill], label=mill,
        alpha=0.75, s=55, zorder=3,
    )

ax_sc.plot(x_line_arc, y_line_arc, color="black", linewidth=1.5, linestyle="--", zorder=4)
ax_sc.axvline(1.0, color="red", linewidth=1.2, linestyle=":", alpha=0.8, label="AR/C = 1")

p_str_arc = "p < 0.001" if p_arc < 0.001 else f"p = {p_arc:.3f}"
ax_sc.annotate(
    f"r = {r_arc:.3f}\n{p_str_arc}\nn = {len(x_arc)}",
    xy=(0.05, 0.90), xycoords="axes fraction",
    fontsize=10, verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
)

ax_sc.set_xlabel("AR/C", fontsize=10)
ax_sc.set_ylabel("Color (IU)", fontsize=10)
ax_sc.legend(loc="upper right", fontsize=8, ncol=2)

plt.tight_layout()
fig_sc.savefig("../outputs/figures/Q7_color/07b_arc_vs_color.png", dpi=150, bbox_inches="tight")
print("Figure saved: Q7_color/07b_arc_vs_color.png")
plt.close(fig_sc)


# -- Figure 3: boxplot per mill (8 subplots) --
fig3, axes3 = plt.subplots(2, 4, figsize=(20, 10))
fig3.suptitle("Color by AR/C group — per mill", fontsize=14, fontweight="bold")
axes3 = axes3.flatten()

for i, mill in enumerate(MILLS):
    ax = axes3[i]
    md = df[df["Ingenio"] == mill]
    g0 = md[md["arc_group"] == "AR/C < 1"]["Color"].values
    g1 = md[md["arc_group"] == "AR/C >= 1"]["Color"].values

    bp = ax.boxplot(
        [g0, g1],
        tick_labels=[f"<1\n(n={len(g0)})", f">=1\n(n={len(g1)})"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], ["#f7a1a1", "#a8d5e2"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    if len(g0) > 1 and len(g1) > 1:
        t, p = stats.ttest_ind(g0, g1, equal_var=False)
        p_label = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.annotate(p_label, xy=(0.5, 0.93), xycoords="axes fraction",
                    fontsize=8, ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

    ax.set_title(mill, fontsize=10, fontweight="bold")
    ax.set_ylabel("Color (IU)", fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
fig3.savefig("../outputs/figures/Q7_color/07c_color_arc_group_per_mill.png", dpi=150, bbox_inches="tight")
print("Figure saved: Q7_color/07c_color_arc_group_per_mill.png")
plt.close(fig3)


# -- Figure 4: scatter AR/C vs Color per mill (8 subplots) --
fig4, axes4 = plt.subplots(2, 4, figsize=(20, 10))
fig4.suptitle("AR/C vs Color — per mill", fontsize=14, fontweight="bold")
axes4 = axes4.flatten()

for i, mill in enumerate(MILLS):
    ax = axes4[i]
    md = df[df["Ingenio"] == mill]
    x = md["AR_Cenizas"].values
    y = md["Color"].values

    ax.scatter(x, y, color=MILL_COLORS[mill], alpha=0.75, s=40, zorder=3)

    if len(x) > 2:
        r, p = stats.pearsonr(x, y)
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.2, linestyle="--")
        p_label = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.annotate(f"r = {r:.3f}\n{p_label}", xy=(0.05, 0.90), xycoords="axes fraction",
                    fontsize=8, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

    ax.axvline(1.0, color="red", linewidth=1, linestyle=":", alpha=0.7)
    ax.set_title(mill, fontsize=10, fontweight="bold")
    ax.set_xlabel("AR/C", fontsize=8)
    ax.set_ylabel("Color (IU)", fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
fig4.savefig("../outputs/figures/Q7_color/07d_arc_vs_color_per_mill.png", dpi=150, bbox_inches="tight")
print("Figure saved: Q7_color/07d_arc_vs_color_per_mill.png")
plt.close(fig4)
