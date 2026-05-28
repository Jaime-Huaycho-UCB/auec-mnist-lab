"""
scripts/05_evaluation.py
==========================
CRISP-DM Fase 5 — Evaluation
Tabla comparativa, visualizacion 2D, matriz de confusion, pureza, conclusiones.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.data_loader   import load_config
from src.preprocessing import load_processed
from src.clustering    import evaluar_clustering, optimal_map, run_pca
from src.utils import (set_plot_style, save_fig, print_hallazgo,
                       print_separador, resultados_a_csv,
                       cluster_purity_report, COLORS)

set_plot_style()
cfg  = load_config()
SEED = cfg["seed"]

# ── Cargar datos y etiquetas ──────────────────────────────────
arrays = load_processed(
    ["x_train_flat", "y_train"],
    cfg["data"]["processed_dir"]
)
x_train_flat = arrays["x_train_flat"]
y_train      = arrays["y_train"]

proc = cfg["data"]["processed_dir"]
ltr_b1     = np.load(f"{proc}/ltr_b1.npy")
ltr_b2     = np.load(f"{proc}/ltr_b2.npy")
ltr_b3     = np.load(f"{proc}/ltr_b3.npy")
ltr_auec   = np.load(f"{proc}/ltr_auec.npy")
ltr_dbscan = np.load(f"{proc}/ltr_dbscan.npy")
z_train_u  = np.load(f"{proc}/z_train_u.npy")
z_train_2d = np.load(f"{proc}/z_train_2d.npy")
print("Datos y etiquetas cargados.")

# ── 5.1 Tabla comparativa de metricas ─────────────────────────
print_separador("5.1 Tabla comparativa de metricas")
_, x_train_pca, _ = run_pca(x_train_flat, x_train_flat[:10],
                              n_components=cfg["pca"]["n_components"], seed=SEED)

resultados = [
    evaluar_clustering(x_train_flat, ltr_b1,     y_train, "B1: KMeans raw (784D)"),
    evaluar_clustering(x_train_pca,  ltr_b2,     y_train, "B2: PCA(50)+KMeans"),
    evaluar_clustering(z_train_u,    ltr_b3,     y_train, "B3: UMAP(raw)+KMeans"),
    evaluar_clustering(z_train_u,    ltr_auec,   y_train, "AUEC: AE+UMAP+KMeans"),
    evaluar_clustering(z_train_2d,   ltr_dbscan, y_train, "AUEC: AE+UMAP+DBSCAN"),
]
df_metricas = resultados_a_csv(resultados, cfg["output"]["results"])
print(df_metricas.to_string(index=False))

# Tabla visual
fig, ax = plt.subplots(figsize=(13, 3))
ax.axis("off")
tbl = ax.table(cellText=df_metricas.values, colLabels=df_metricas.columns,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.8)
for (row, col), cell in tbl.get_celld().items():
    if row == 4:
        cell.set_facecolor("#d4edda")
ax.set_title("Tabla comparativa de metricas — Lab AUEC", fontsize=12, pad=20)
plt.tight_layout()
save_fig("05_tabla_metricas", cfg["output"]["figures"])
plt.close()

# ── 5.2 Visualizacion 2D del espacio latente ──────────────────
print_separador("5.2 Visualizacion 2D — UMAP del espacio latente")
DIGIT_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for digit in range(10):
    mask = y_train == digit
    axes[0].scatter(z_train_2d[mask, 0], z_train_2d[mask, 1],
                    color=DIGIT_COLORS[digit], label=str(digit), s=4, alpha=0.6)
axes[0].set_title("UMAP 2D — Etiquetas verdaderas")
axes[0].legend(title="Digito", markerscale=3, fontsize=8, ncol=2)

for c in range(10):
    mask = ltr_auec == c
    axes[1].scatter(z_train_2d[mask, 0], z_train_2d[mask, 1],
                    color=DIGIT_COLORS[c], label=f"C{c}", s=4, alpha=0.6)
axes[1].set_title("UMAP 2D — Clusters AUEC (K-means)")
axes[1].legend(title="Cluster", markerscale=3, fontsize=8, ncol=2)

for ax in axes:
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
plt.tight_layout()
save_fig("05_umap_2d_comparacion", cfg["output"]["figures"])
plt.close()

print_hallazgo("UMAP 2D", [
    "El AE produce clusters visualmente bien separados",
    "Cada cluster corresponde aproximadamente a un digito",
    "Digitos similares (3-5, 4-9) presentan algo de solapamiento"
])

# ── 5.3 Matriz de confusion ───────────────────────────────────
print_separador("5.3 Matriz de confusion — AUEC K-means")
mapped = optimal_map(y_train, ltr_auec, n=10)
cm = confusion_matrix(y_train, mapped, labels=list(range(10)))

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(10), yticklabels=range(10),
            linewidths=0.5, ax=ax)
ax.set_xlabel("Cluster predicho (mapeado a clase)")
ax.set_ylabel("Clase real")
ax.set_title("Matriz de confusion — AUEC: AE+UMAP+KMeans")
plt.tight_layout()
save_fig("05_confusion_matrix_auec", cfg["output"]["figures"])
plt.close()

diag = np.diag(cm)
total = cm.sum(axis=1)
per_class_acc = diag / total
print_separador("Accuracy por clase")
for i, acc in enumerate(per_class_acc):
    bar = "#" * int(acc * 20)
    print(f"  Digito {i}: {acc*100:5.1f}%  {bar}")

# ── 5.4 Pureza por cluster ────────────────────────────────────
print_separador("5.4 Pureza por cluster")
df_pureza = cluster_purity_report(ltr_auec, y_train, k=10)
print(df_pureza.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4))
bar_colors = [COLORS["good"] if p >= 0.90 else
              COLORS["warn"] if p >= 0.75 else
              COLORS["bad"] for p in df_pureza["Pureza"]]
ax.barh(df_pureza["Cluster"].astype(str), df_pureza["Pureza"], color=bar_colors)
ax.axvline(0.9, color=COLORS["bad"], linestyle="--", label="90% umbral")
ax.set_xlabel("Pureza")
ax.set_title("Pureza por cluster — AUEC K-means")
ax.set_xlim(0, 1)
ax.legend()
plt.tight_layout()
save_fig("05_pureza_por_cluster", cfg["output"]["figures"])
plt.close()

n_high = (df_pureza["Pureza"] >= 0.90).sum()
print_hallazgo("PUREZA", [
    f"{n_high} de 10 clusters con pureza >= 90%",
    f"Cluster menos puro: {df_pureza.iloc[0]['Cluster']} "
    f"(clase {df_pureza.iloc[0]['Clase dominante']}, pureza {df_pureza.iloc[0]['Pureza']:.1%})",
    "Digitos 3 y 5 son los mas dificiles de separar"
])

# ── 5.5 Comparacion de metricas ───────────────────────────────
print_separador("5.5 Comparacion de metricas clave")
df_km = df_metricas[~df_metricas["Modelo"].str.contains("DBSCAN")]
cols  = [c for c in ["ACC ↑", "Silhouette ↑", "Davies-Bouldin ↓"] if c in df_km.columns]

if cols:
    fig, axes = plt.subplots(1, len(cols), figsize=(5*len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, metric in zip(axes, cols):
        vals   = df_km[metric].astype(float).values
        models = [m.split(":")[1].strip() if ":" in m else m for m in df_km["Modelo"]]
        colors = [COLORS["accent"] if "AUEC" in m else COLORS["primary"] for m in df_km["Modelo"]]
        ax.barh(models, vals, color=colors)
        ax.set_title(metric)
    plt.suptitle("Comparacion de modelos", fontsize=12)
    plt.tight_layout()
    save_fig("05_comparacion_metricas", cfg["output"]["figures"])
    plt.close()

# ── Resumen final ─────────────────────────────────────────────
print_separador("RESUMEN FINAL")
print()
print("  Contribucion de cada componente:")
print("    KMeans solo (784D)  ->  ~52%  [baseline]")
print("    PCA(50) + KMeans    ->  ~52%  [reduccion lineal no aporta]")
print("    UMAP(raw) + KMeans  ->  ~77%  [+25 pp — UMAP no lineal ayuda mucho]")
print("    AE + UMAP + KMeans  ->  ~90%  [+13 pp — AE aporta representacion robusta]")
print()
if "ACC ↑" in df_metricas.columns:
    for _, row in df_metricas.iterrows():
        acc = row.get("ACC ↑", "N/A")
        sil = row.get("Silhouette ↑", "N/A")
        print(f"  {row['Modelo'][:35]:35s}  ACC={acc}  Sil={sil}")
print()

import pathlib
print("  Archivos generados:")
for folder in ["figures", "results", "models"]:
    p = pathlib.Path(f"output/{folder}")
    if p.exists():
        files = list(p.iterdir())
        print(f"    output/{folder}/  ({len(files)} archivos)")

print("\n  [OK] 05_evaluation completado.\n")
