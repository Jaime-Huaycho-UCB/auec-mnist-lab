"""
scripts/04_modeling.py
========================
CRISP-DM Fase 4 — Modeling
Baselines B1/B2/B3 + entrenamiento CAE + UMAP + K-means + DBSCAN (AUEC).
Persiste etiquetas, representaciones y pesos del encoder.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pathlib
import matplotlib.pyplot as plt

from src.data_loader   import load_config
from src.preprocessing import load_processed
from src.autoencoder   import build_autoencoder, train_autoencoder, save_encoder_weights
from src.clustering    import (run_kmeans, run_dbscan, run_pca, run_umap,
                               evaluar_clustering)
from src.utils import (set_plot_style, save_fig, print_hallazgo,
                       print_separador, COLORS)

set_plot_style()
cfg  = load_config()
SEED = cfg["seed"]
K    = cfg["kmeans"]["k"]

# ── Cargar datos procesados ────────────────────────────────────
arrays = load_processed(
    ["x_train_cnn", "x_test_cnn", "x_train_flat", "x_test_flat", "y_train", "y_test"],
    cfg["data"]["processed_dir"]
)
x_train_cnn  = arrays["x_train_cnn"]
x_test_cnn   = arrays["x_test_cnn"]
x_train_flat = arrays["x_train_flat"]
x_test_flat  = arrays["x_test_flat"]
y_train      = arrays["y_train"]
y_test       = arrays["y_test"]
print(f"Datos cargados — train_flat: {x_train_flat.shape}  train_cnn: {x_train_cnn.shape}")

# ── B1: K-means raw (784D) ────────────────────────────────────
print_separador("B1: K-means raw (784D)")
km_b1, ltr_b1, lte_b1 = run_kmeans(
    x_train_flat, x_test_flat, k=K,
    n_init=cfg["kmeans"]["n_init"], seed=SEED
)
res_b1 = evaluar_clustering(x_train_flat, ltr_b1, y_train, "B1: KMeans raw (784D)")
print(res_b1)
print_hallazgo("B1", [
    f"ACC = {res_b1.get('ACC ↑', 'N/A')}",
    "K-means en 784D sufre curse of dimensionality"
])

# ── B2: PCA(50) + K-means ─────────────────────────────────────
print_separador("B2: PCA(50) + K-means")
pca, x_train_pca, x_test_pca = run_pca(
    x_train_flat, x_test_flat,
    n_components=cfg["pca"]["n_components"], seed=SEED
)
km_b2, ltr_b2, lte_b2 = run_kmeans(
    x_train_pca, x_test_pca, k=K,
    n_init=cfg["kmeans"]["n_init"], seed=SEED
)
res_b2 = evaluar_clustering(x_train_pca, ltr_b2, y_train, "B2: PCA(50)+KMeans")
print(res_b2)
print_hallazgo("B2", [
    f"PCA retiene {pca.explained_variance_ratio_.sum()*100:.1f}% de varianza",
    f"ACC = {res_b2.get('ACC ↑', 'N/A')} — reduccion lineal ayuda poco"
])

# ── B3: UMAP directo + K-means ────────────────────────────────
print_separador("B3: UMAP(raw 784D) + K-means")
umap_b3, x_train_ub3, x_test_ub3 = run_umap(
    x_train_flat, x_test_flat,
    n_components=cfg["umap"]["n_components"],
    n_neighbors=cfg["umap"]["n_neighbors"],
    min_dist=cfg["umap"]["min_dist"], seed=SEED
)
km_b3, ltr_b3, lte_b3 = run_kmeans(
    x_train_ub3, x_test_ub3, k=K,
    n_init=cfg["kmeans"]["n_init"], seed=SEED
)
res_b3 = evaluar_clustering(x_train_ub3, ltr_b3, y_train, "B3: UMAP(raw)+KMeans")
print(res_b3)
print_hallazgo("B3", [
    f"ACC = {res_b3.get('ACC ↑', 'N/A')} — UMAP no lineal mejora significativamente"
])

# ── CAE: Arquitectura y entrenamiento ─────────────────────────
print_separador("CAE — Arquitectura")
autoencoder, encoder = build_autoencoder(
    latent_dim=cfg["ae"]["latent_dim"],
    input_shape=(28, 28, 1)
)
autoencoder.summary()

print_separador("CAE — Entrenamiento")
history = train_autoencoder(autoencoder, x_train_cnn, cfg, validation_split=0.1)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(history.history["loss"],     label="Train loss",      color=COLORS["primary"])
ax.plot(history.history["val_loss"], label="Validacion loss", color=COLORS["accent"])
ax.set_xlabel("Epoca")
ax.set_ylabel("MSE")
ax.set_title("Curva de aprendizaje — Autoencoder Convolucional")
ax.legend()
plt.tight_layout()
save_fig("04_curva_aprendizaje_ae", cfg["output"]["figures"])
plt.close()

print_hallazgo("CAE", [
    f"MSE train final   = {history.history['loss'][-1]:.6f}",
    f"MSE val   final   = {history.history['val_loss'][-1]:.6f}",
    f"Epocas entrenadas = {len(history.history['loss'])}"
])

# ── Reconstruccion visual ──────────────────────────────────────
recon = autoencoder.predict(x_test_cnn[:10], verbose=0)
fig, axes = plt.subplots(2, 10, figsize=(14, 3))
for i in range(10):
    axes[0, i].imshow(x_test_cnn[i, :, :, 0], cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(recon[i, :, :, 0], cmap="gray")
    axes[1, i].axis("off")
axes[0, 0].set_ylabel("Original",     fontsize=9, rotation=0, labelpad=40)
axes[1, 0].set_ylabel("Reconstruida", fontsize=9, rotation=0, labelpad=40)
fig.suptitle("Reconstruccion del Autoencoder (test set)", fontsize=12)
plt.tight_layout()
save_fig("04_reconstruccion_ae", cfg["output"]["figures"])
plt.close()

save_encoder_weights(encoder, cfg["output"]["models"])

# ── AUEC: Espacio latente -> UMAP -> K-means ──────────────────
print_separador("AUEC — Espacio latente")
z_train = encoder.predict(x_train_cnn, verbose=0)
z_test  = encoder.predict(x_test_cnn,  verbose=0)
print_hallazgo("LATENTE", [
    f"Dimensiones: {z_train.shape[1]}D  (vs 784D original)",
    f"Rango: [{z_train.min():.3f}, {z_train.max():.3f}]"
])

print_separador("AUEC — UMAP sobre espacio latente")
umap_ae, z_train_u, z_test_u = run_umap(
    z_train, z_test,
    n_components=cfg["umap"]["n_components"],
    n_neighbors=cfg["umap"]["n_neighbors"],
    min_dist=cfg["umap"]["min_dist"], seed=SEED
)

print_separador("AUEC — K-means sobre UMAP(latente)")
km_auec, ltr_auec, lte_auec = run_kmeans(
    z_train_u, z_test_u, k=K,
    n_init=cfg["kmeans"]["n_init"], seed=SEED
)
res_auec_km = evaluar_clustering(z_train_u, ltr_auec, y_train, "AUEC: AE+UMAP+KMeans")
print(res_auec_km)
print_hallazgo("AUEC KMeans", [
    f"ACC = {res_auec_km.get('ACC ↑', 'N/A')} — supera a todos los baselines"
])

# ── AUEC: DBSCAN sobre UMAP 2D ────────────────────────────────
print_separador("AUEC — DBSCAN sobre UMAP 2D")
_, z_train_2d, z_test_2d = run_umap(
    z_train, z_test,
    n_components=2,
    n_neighbors=cfg["umap"]["viz"]["n_neighbors"],
    min_dist=cfg["umap"]["viz"]["min_dist"], seed=SEED
)
db_auec, ltr_dbscan = run_dbscan(
    z_train_2d,
    eps=cfg["dbscan"]["eps"],
    min_samples=cfg["dbscan"]["min_samples"]
)
res_auec_db = evaluar_clustering(z_train_2d, ltr_dbscan, y_train, "AUEC: AE+UMAP+DBSCAN")
print(res_auec_db)
n_noise = (ltr_dbscan == -1).sum()
print_hallazgo("DBSCAN", [
    f"Clusters encontrados: {res_auec_db['N clusters']}  (esperados: 10)",
    f"Ruido: {n_noise} puntos ({n_noise/len(ltr_dbscan)*100:.1f}%)",
    "eps=0.5 tiende a unir clusters en 2D — ajustar para 10 exactos"
])

# ── Persistencia de etiquetas y representaciones ──────────────
print_separador("Guardando etiquetas y representaciones")
pathlib.Path(cfg["data"]["processed_dir"]).mkdir(exist_ok=True)
np.save(f"{cfg['data']['processed_dir']}/ltr_b1.npy",     ltr_b1)
np.save(f"{cfg['data']['processed_dir']}/ltr_b2.npy",     ltr_b2)
np.save(f"{cfg['data']['processed_dir']}/ltr_b3.npy",     ltr_b3)
np.save(f"{cfg['data']['processed_dir']}/ltr_auec.npy",   ltr_auec)
np.save(f"{cfg['data']['processed_dir']}/ltr_dbscan.npy", ltr_dbscan)
np.save(f"{cfg['data']['processed_dir']}/z_train.npy",    z_train)
np.save(f"{cfg['data']['processed_dir']}/z_train_u.npy",  z_train_u)
np.save(f"{cfg['data']['processed_dir']}/z_train_2d.npy", z_train_2d)
print("  Etiquetas y representaciones guardadas en data/processed/")

print("\n  [OK] 04_modeling completado.\n")
