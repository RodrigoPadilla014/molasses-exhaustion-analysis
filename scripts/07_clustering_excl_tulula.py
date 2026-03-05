import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# data — exclude Tulula (structural outlier, No Mielera every season)
df = pd.read_excel("../data/processed/Base maestra.xlsx")
df["Mes"] = df["Mes"].str.strip()
df["DPO"] = df["Pureza_Real"] - df["Pureza_Obj"]

df = df[df["Ingenio"] != "Tulula"].copy()
print(f"Mills included: {sorted(df['Ingenio'].unique())}")
print(f"Total samples after excluding Tulula: {len(df)}")

MILLS  = sorted(df["Ingenio"].unique())
ZAFRAS = sorted(df["Zafra"].unique())
MILL_COLORS  = {m: c for m, c in zip(MILLS, plt.cm.tab10.colors)}

os.makedirs("../outputs/figures/Q10_clustering_excl_tulula", exist_ok=True)

# same features as original clustering
FEATURES = ["AR_Cenizas", "Pureza_Real", "DPO", "Color", "kg/t_MF"]

df_cluster = df[["Ingenio", "Zafra", "Mes"] + FEATURES].dropna(subset=FEATURES).copy()
print(f"Samples used for clustering: {len(df_cluster)}")

# standardize
X = df_cluster[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ── PCA ────────────────────────────────────────────────────────────────────────

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained  = pca.explained_variance_ratio_
cumulative = explained.cumsum()

print("\nPCA explained variance:")
for i, (e, c) in enumerate(zip(explained, cumulative)):
    print(f"  PC{i+1}: {e:.3f}  cumulative: {c:.3f}")

loadings = pd.DataFrame(
    pca.components_.T,
    index=FEATURES,
    columns=[f"PC{i+1}" for i in range(len(FEATURES))]
).round(3)
print("\nPCA loadings:")
print(loadings.to_string())

# -- Figure 1: explained variance --
fig1, ax1 = plt.subplots(figsize=(7, 5))
fig1.suptitle("PCA explained variance (excl. Tulula)", fontsize=13, fontweight="bold")
ax1.bar(range(1, len(explained)+1), explained, alpha=0.7, color="steelblue", label="Individual")
ax1.plot(range(1, len(cumulative)+1), cumulative, marker="o", color="red", linewidth=1.5, label="Cumulative")
ax1.axhline(0.80, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="80% threshold")
ax1.set_xlabel("Principal component", fontsize=10)
ax1.set_ylabel("Explained variance ratio", fontsize=10)
ax1.legend(fontsize=8)
plt.tight_layout()
fig1.savefig("../outputs/figures/Q10_clustering_excl_tulula/10a_pca_variance.png", dpi=150, bbox_inches="tight")
print("\nFigure saved: Q10_clustering_excl_tulula/10a_pca_variance.png")
plt.close(fig1)


# ── ELBOW + SILHOUETTE — choose k ─────────────────────────────────────────────

inertias    = []
silhouettes = []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Choosing number of clusters (excl. Tulula)", fontsize=13, fontweight="bold")

ax2a.plot(K_range, inertias, marker="o", linewidth=1.5, color="steelblue")
ax2a.set_xlabel("Number of clusters (k)", fontsize=10)
ax2a.set_ylabel("Inertia", fontsize=10)
ax2a.set_title("Elbow method", fontsize=10)

ax2b.plot(K_range, silhouettes, marker="o", linewidth=1.5, color="darkorange")
ax2b.set_xlabel("Number of clusters (k)", fontsize=10)
ax2b.set_ylabel("Silhouette score", fontsize=10)
ax2b.set_title("Silhouette score", fontsize=10)

plt.tight_layout()
fig2.savefig("../outputs/figures/Q10_clustering_excl_tulula/10b_elbow_silhouette.png", dpi=150, bbox_inches="tight")
print("Figure saved: Q10_clustering_excl_tulula/10b_elbow_silhouette.png")
plt.close(fig2)

print("\nSilhouette scores:")
for k, s in zip(K_range, silhouettes):
    print(f"  k={k}: {s:.3f}")

print("\n-- Review 10a and 10b, then set K below and re-run to generate final clustering --")

# ── FINAL CLUSTERING — set K after reviewing elbow/silhouette above ────────────
# Uncomment and set K once you have decided:

K = 3

CLUSTER_NAMES  = {0: "Normal", 1: "Mielera", 2: "No Mielera"}
CLUSTER_COLORS = {0: "#4CAF50", 1: "#F44336", 2: "#FF9800"}

km_final = KMeans(n_clusters=K, random_state=42, n_init=10)
df_cluster["cluster"] = km_final.fit_predict(X_scaled)

# -- PCA scatter by cluster --
fig3, ax3 = plt.subplots(figsize=(9, 7))
fig3.suptitle(f"PCA scatter — k={K} clusters (excl. Tulula)", fontsize=13, fontweight="bold")
for c in range(K):
    mask = df_cluster["cluster"] == c
    ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
                color=CLUSTER_COLORS[c], label=CLUSTER_NAMES[c],
                alpha=0.75, s=55, zorder=3)
ax3.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)", fontsize=10)
ax3.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)", fontsize=10)
ax3.legend(fontsize=9)
plt.tight_layout()
fig3.savefig("../outputs/figures/Q10_clustering_excl_tulula/10c_pca_clusters.png", dpi=150, bbox_inches="tight")
print("Figure saved: Q10_clustering_excl_tulula/10c_pca_clusters.png")
plt.close(fig3)

# -- PCA scatter by mill --
fig4, ax4 = plt.subplots(figsize=(9, 7))
fig4.suptitle("PCA scatter — colored by mill (excl. Tulula)", fontsize=13, fontweight="bold")
for mill in MILLS:
    mask = df_cluster["Ingenio"] == mill
    ax4.scatter(X_pca[mask, 0], X_pca[mask, 1],
                color=MILL_COLORS[mill], label=mill,
                alpha=0.75, s=55, zorder=3)
ax4.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)", fontsize=10)
ax4.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)", fontsize=10)
ax4.legend(fontsize=8, ncol=2)
plt.tight_layout()
fig4.savefig("../outputs/figures/Q10_clustering_excl_tulula/10d_pca_by_mill.png", dpi=150, bbox_inches="tight")
print("Figure saved: Q10_clustering_excl_tulula/10d_pca_by_mill.png")
plt.close(fig4)

# -- Cluster profile heatmap --
cluster_means = df_cluster.groupby("cluster")[FEATURES].mean()
cluster_means_scaled = pd.DataFrame(
    scaler.transform(cluster_means),
    index=cluster_means.index,
    columns=FEATURES
)
fig5, ax5 = plt.subplots(figsize=(9, 4))
fig5.suptitle("Cluster profiles (standardized means, excl. Tulula)", fontsize=13, fontweight="bold")
im = ax5.imshow(cluster_means_scaled.values, aspect="auto", cmap="RdYlGn")
ax5.set_xticks(range(len(FEATURES)))
ax5.set_xticklabels(FEATURES, fontsize=9)
ax5.set_yticks(range(K))
ax5.set_yticklabels([CLUSTER_NAMES[i] for i in range(K)], fontsize=9)
plt.colorbar(im, ax=ax5, label="Standardized value")
for i in range(K):
    for j, feat in enumerate(FEATURES):
        ax5.text(j, i, f"{cluster_means.iloc[i, j]:.2f}",
                 ha="center", va="center", fontsize=8, color="black")
plt.tight_layout()
fig5.savefig("../outputs/figures/Q10_clustering_excl_tulula/10e_cluster_heatmap.png", dpi=150, bbox_inches="tight")
print("Figure saved: Q10_clustering_excl_tulula/10e_cluster_heatmap.png")
plt.close(fig5)

# -- Cluster composition --
print(f"\n-- Cluster composition by mill (k={K}) --")
print(pd.crosstab(df_cluster["Ingenio"], df_cluster["cluster"], colnames=["Cluster"]).to_string())
print(f"\n-- Cluster composition by zafra (k={K}) --")
print(pd.crosstab(df_cluster["Zafra"], df_cluster["cluster"], colnames=["Cluster"]).to_string())
print(f"\n-- Cluster means (raw) --")
print(df_cluster.groupby("cluster")[FEATURES].mean().round(3).to_string())

# -- Export --
sample_list = df_cluster[["Ingenio", "Zafra", "Mes", "cluster"] + FEATURES].copy()
sample_list["cluster"] = sample_list["cluster"] + 1
by_mill       = pd.crosstab(df_cluster["Ingenio"], df_cluster["cluster"] + 1, colnames=["Cluster"])
by_zafra_mill = pd.crosstab([df_cluster["Zafra"], df_cluster["Ingenio"]], df_cluster["cluster"] + 1, colnames=["Cluster"])
means_table   = df_cluster.groupby("cluster")[FEATURES].mean()
means_table.index = [CLUSTER_NAMES[i] for i in means_table.index]

with pd.ExcelWriter("../outputs/cluster_tables_excl_tulula.xlsx", engine="openpyxl") as writer:
    sample_list.to_excel(writer, sheet_name="Sample list", index=False)
    by_mill.to_excel(writer, sheet_name="By mill")
    by_zafra_mill.to_excel(writer, sheet_name="By zafra x mill")
    means_table.round(3).to_excel(writer, sheet_name="Cluster means")
print("\nTables exported: outputs/cluster_tables_excl_tulula.xlsx")

df_cluster[["Ingenio", "Zafra", "Mes", "cluster"]].to_csv("../outputs/cluster_labels_excl_tulula.csv", index=False)
print("Cluster labels exported: outputs/cluster_labels_excl_tulula.csv")


# ── CLUSTER TRAJECTORY HEATMAP ─────────────────────────────────────────────────

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ZAFRAS = sorted(df_cluster["Zafra"].unique())

trajectory = (
    df_cluster.groupby(["Ingenio", "Zafra"])["cluster"]
    .agg(lambda x: x.mode()[0])
    .reset_index()
)
# map cluster number to display label (1=Normal, 2=Mielera, 3=No Mielera)
DISPLAY_ORDER = {0: 1, 1: 2, 2: 3}  # Normal=1(green), Mielera=2(red), No Mielera=3(orange)
trajectory["cluster_display"] = trajectory["cluster"].map(DISPLAY_ORDER)

pivot = trajectory.pivot(index="Ingenio", columns="Zafra", values="cluster_display")

# green=Normal, red=Mielera, orange=No Mielera
cmap = ListedColormap(["#4CAF50", "#F44336", "#FF9800"])

fig_traj, ax_traj = plt.subplots(figsize=(10, 5))
fig_traj.suptitle("Cluster trajectory per mill across zafras (excl. Tulula)", fontsize=13, fontweight="bold")

im = ax_traj.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0.5, vmax=3.5)

ax_traj.set_xticks(range(len(pivot.columns)))
ax_traj.set_xticklabels(pivot.columns, fontsize=9, rotation=30)
ax_traj.set_yticks(range(len(pivot.index)))
ax_traj.set_yticklabels(pivot.index, fontsize=9)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.iloc[i, j]
        label = {1: "N", 2: "M", 3: "NM"}[int(val)]
        ax_traj.text(j, i, label, ha="center", va="center", fontsize=11, fontweight="bold", color="white")

legend_elements = [
    Patch(facecolor="#4CAF50", label="Normal"),
    Patch(facecolor="#F44336", label="Mielera"),
    Patch(facecolor="#FF9800", label="No Mielera"),
]
ax_traj.legend(handles=legend_elements, loc="upper right", fontsize=8, bbox_to_anchor=(1.3, 1))

plt.tight_layout()
fig_traj.savefig("../outputs/figures/Q10_clustering_excl_tulula/10f_cluster_trajectory.png", dpi=150, bbox_inches="tight")
print("Figure saved: Q10_clustering_excl_tulula/10f_cluster_trajectory.png")
plt.close(fig_traj)

print("\n-- Cluster trajectory per mill (mode per zafra) --")
print(trajectory.pivot(index="Ingenio", columns="Zafra", values="cluster_display").to_string())
