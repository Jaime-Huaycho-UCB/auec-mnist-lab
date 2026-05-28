"""
scripts/03_data_preparation.py
================================
CRISP-DM Fase 3 — Data Preparation
Crea subsets reproducibles, normaliza, reshape y persiste en data/processed/.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pathlib
import matplotlib.pyplot as plt

from src.data_loader   import load_config, load_mnist
from src.preprocessing import (create_subsets, normalize, reshape_cnn,
                                reshape_flat, save_processed)
from src.utils import set_plot_style, save_fig, print_hallazgo, print_separador, COLORS

set_plot_style()
cfg  = load_config()
SEED = cfg["seed"]

# ── 3.1 Carga ────────────────────────────────────────────────
print_separador("3.1 Carga de datos crudos")
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_mnist(cfg["data"]["raw_dir"])
print(f"  train: {x_train_raw.shape}  test: {x_test_raw.shape}")

# ── 3.2 Subsets reproducibles ─────────────────────────────────
print_separador("3.2 Creacion de subsets reproducibles")
N_TRAIN = cfg["data"]["n_train"]
N_TEST  = cfg["data"]["n_test"]

x_train, y_train, x_test, y_test = create_subsets(
    x_train_raw, y_train_raw,
    x_test_raw,  y_test_raw,
    n_train=N_TRAIN, n_test=N_TEST, seed=SEED
)
print(f"  Train: {x_train.shape}  ({N_TRAIN/60000*100:.0f}% del train original)")
print(f"  Test : {x_test.shape}   ({N_TEST/10000*100:.0f}% del test original)")

vals, counts = np.unique(y_train, return_counts=True)
print("\n  Distribucion train subset:")
for v, c in zip(vals, counts):
    print(f"    Digito {v}: {c:4d}  ({c/N_TRAIN*100:.1f}%)")

# ── 3.3 Normalizacion ─────────────────────────────────────────
print_separador("3.3 Normalizacion [0, 255] -> [0.0, 1.0]")
x_train_norm = normalize(x_train)
x_test_norm  = normalize(x_test)
print(f"  Antes  : dtype={x_train.dtype}  rango=[{x_train.min()}, {x_train.max()}]")
print(f"  Despues: dtype={x_train_norm.dtype}  rango=[{x_train_norm.min():.4f}, {x_train_norm.max():.4f}]")

fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for i in range(10):
    axes[i//5, i%5].imshow(x_train_norm[i], cmap="gray", vmin=0, vmax=1)
    axes[i//5, i%5].set_title(f"Label: {y_train[i]}")
    axes[i//5, i%5].axis("off")
fig.suptitle("Muestras normalizadas del subset de train", fontsize=13)
plt.tight_layout()
save_fig("03_muestras_normalizadas", cfg["output"]["figures"])
plt.close()

# ── 3.4 Reshape ───────────────────────────────────────────────
print_separador("3.4 Reshape: CNN (N,28,28,1) y Flat (N,784)")
x_train_cnn  = reshape_cnn(x_train_norm)
x_test_cnn   = reshape_cnn(x_test_norm)
x_train_flat = reshape_flat(x_train_norm)
x_test_flat  = reshape_flat(x_test_norm)

print(f"  x_train_cnn  : {x_train_cnn.shape}   (input Autoencoder)")
print(f"  x_test_cnn   : {x_test_cnn.shape}")
print(f"  x_train_flat : {x_train_flat.shape}  (input baselines)")
print(f"  x_test_flat  : {x_test_flat.shape}")

# ── 3.5 Verificacion de calidad ───────────────────────────────
print_separador("3.5 Verificacion de calidad")
assert x_train_cnn.min()  >= 0.0 and x_train_cnn.max()  <= 1.0
assert x_train_flat.min() >= 0.0 and x_train_flat.max() <= 1.0
assert x_train_cnn.shape  == (N_TRAIN, 28, 28, 1)
assert x_train_flat.shape == (N_TRAIN, 784)
assert not np.isnan(x_train_cnn).any()
assert not np.isnan(x_train_flat).any()
print_hallazgo("CALIDAD", [
    "Todos los assertions pasaron.",
    "Rango [0,1], shapes correctos, sin NaN."
])

# ── 3.6 Persistencia ──────────────────────────────────────────
print_separador("3.6 Persistencia en data/processed/")
arrays = {
    "x_train_cnn" : x_train_cnn,
    "x_test_cnn"  : x_test_cnn,
    "x_train_flat": x_train_flat,
    "x_test_flat" : x_test_flat,
    "y_train"     : y_train,
    "y_test"      : y_test,
}
save_processed(arrays, cfg["data"]["processed_dir"])

for f in sorted(pathlib.Path(cfg["data"]["processed_dir"]).glob("*.npy")):
    print(f"    {f.name}  ({f.stat().st_size/1e6:.2f} MB)")

print("\n  [OK] 03_data_preparation completado.\n")
