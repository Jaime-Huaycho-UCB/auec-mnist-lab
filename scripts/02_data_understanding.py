"""
scripts/02_data_understanding.py
=================================
CRISP-DM Fase 2 — Data Understanding
Carga MNIST, EDA, distribucion de clases, estadisticas de pixeles, varianza PCA.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.data_loader import load_config, load_mnist
from src.utils import set_plot_style, save_fig, print_hallazgo, print_separador, COLORS

set_plot_style()
cfg  = load_config()
SEED = cfg["seed"]

# ── 2.1 Carga ────────────────────────────────────────────────
print_separador("2.1 Carga del dataset MNIST")
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_mnist(cfg["data"]["raw_dir"])
print(f"  Train : {x_train_raw.shape}  dtype={x_train_raw.dtype}  rango=[{x_train_raw.min()}, {x_train_raw.max()}]")
print(f"  Test  : {x_test_raw.shape}   dtype={x_test_raw.dtype}")
print(f"  Clases: {sorted(np.unique(y_train_raw))}")

# ── 2.2 Distribucion de clases ────────────────────────────────
print_separador("2.2 Distribucion de clases")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (data, label) in zip(axes, [(y_train_raw, "Train (60 000)"), (y_test_raw, "Test (10 000)")]):
    vals, counts = np.unique(data, return_counts=True)
    ax.bar(vals, counts, color=COLORS["primary"], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Digito")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribucion de clases — {label}")
    ax.set_xticks(range(10))
    for v, c in zip(vals, counts):
        ax.text(v, c + 50, str(c), ha="center", va="bottom", fontsize=8)
plt.tight_layout()
save_fig("02_distribucion_clases", cfg["output"]["figures"])
plt.close()

print_hallazgo("BALANCE", [
    "MNIST esta casi perfectamente balanceado (~6 000 muestras por clase).",
    "No se necesita sobremuestreo ni pesos de clase."
])

# ── 2.3 Muestras por clase ────────────────────────────────────
print_separador("2.3 Muestras representativas por clase")
fig, axes = plt.subplots(10, 8, figsize=(10, 14))
for digit in range(10):
    idx = np.where(y_train_raw == digit)[0][:8]
    for col, i in enumerate(idx):
        axes[digit, col].imshow(x_train_raw[i], cmap="gray")
        axes[digit, col].axis("off")
    axes[digit, 0].set_ylabel(str(digit), fontsize=11, rotation=0, labelpad=15, va="center")
fig.suptitle("Muestras por clase (8 ejemplos x 10 digitos)", fontsize=13)
plt.tight_layout()
save_fig("02_muestras_por_clase", cfg["output"]["figures"])
plt.close()

# ── 2.4 Estadisticas de pixeles ───────────────────────────────
print_separador("2.4 Estadisticas de pixeles")
means = x_train_raw.mean(axis=0)
stds  = x_train_raw.std(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].imshow(means, cmap="hot")
axes[0].set_title("Media de pixel por posicion")
plt.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(stds, cmap="Blues")
axes[1].set_title("Desviacion estandar por posicion")
plt.colorbar(im1, ax=axes[1])
plt.tight_layout()
save_fig("02_estadisticas_pixeles", cfg["output"]["figures"])
plt.close()

pct_zeros = (x_train_raw == 0).mean() * 100
print_hallazgo("DISPERSION", [
    f"{pct_zeros:.1f}% de los pixeles son 0 (fondo negro)",
    "Alta dispersidad -> justifica reduccion de dimensionalidad",
    "Rango [0, 255] -> normalizacion obligatoria a [0, 1]"
])

# ── 2.5 Varianza explicada PCA ────────────────────────────────
print_separador("2.5 Varianza explicada — justificacion de PCA")
x_flat = x_train_raw[:5000].reshape(-1, 784).astype("float32") / 255.0
pca_full = PCA(n_components=100, random_state=SEED).fit(x_flat)
cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, 101), cumvar, color=COLORS["primary"], linewidth=2)
ax.axhline(85, color=COLORS["warn"], linestyle="--", label="85% varianza")
ax.axvline(50, color=COLORS["accent"], linestyle="--", label="50 componentes")
ax.set_xlabel("Numero de componentes PCA")
ax.set_ylabel("Varianza acumulada (%)")
ax.set_title("Varianza explicada acumulada (muestra 5 000)")
ax.legend()
plt.tight_layout()
save_fig("02_varianza_pca", cfg["output"]["figures"])
plt.close()

idx_85 = int(np.searchsorted(cumvar, 85)) + 1
print_hallazgo("PCA", [
    f"50 componentes retienen aprox. {cumvar[49]:.1f}% de la varianza",
    f"Se necesitan {idx_85} componentes para 85%",
    "-> PCA(50) es razonable para baseline B2"
])

print("\n  [OK] 02_data_understanding completado.\n")
