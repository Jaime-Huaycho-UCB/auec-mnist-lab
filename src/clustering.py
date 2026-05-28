"""
src/clustering.py
=================
Funciones de clustering, reducción de dimensionalidad y métricas.
CRISP-DM: Modeling + Evaluation
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


# ── Métricas ─────────────────────────────────────────────────

def clustering_accuracy(y_true, y_pred):
    """Accuracy óptima via asignación húngara (linear_sum_assignment)."""
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    n = max(y_true.max(), y_pred.max()) + 1
    cost = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cost[p, t] += 1
    row, col = linear_sum_assignment(cost.max() - cost)
    return cost[row, col].sum() / len(y_true)


def evaluar_clustering(X, labels, y_true=None, nombre="", sample_size=5000, seed=42):
    """
    Calcula métricas internas y externas de clustering.

    Métricas internas (no requieren etiquetas reales):
      - Silhouette score  (↑ mejor, rango [-1, 1])
      - Davies-Bouldin    (↓ mejor, ≥ 0)

    Métricas externas (requieren y_true):
      - ACC  via Hungarian algorithm
      - NMI  Normalized Mutual Information
      - ARI  Adjusted Rand Index
    """
    labels = np.array(labels)
    mask   = labels != -1          # excluir ruido de DBSCAN
    X_c, l_c = X[mask], labels[mask]
    n_clusters = len(np.unique(l_c))

    # Silhouette sobre muestra para velocidad en alta dimensión
    if len(X_c) > sample_size:
        idx    = np.random.default_rng(seed).choice(len(X_c), sample_size, replace=False)
        Xs, ls = X_c[idx], l_c[idx]
    else:
        Xs, ls = X_c, l_c

    sil = silhouette_score(Xs, ls, random_state=seed) if n_clusters > 1 else float("nan")
    db  = davies_bouldin_score(X_c, l_c)              if n_clusters > 1 else float("nan")

    resultado = {
        "Modelo":            nombre,
        "N clusters":        n_clusters,
        "Silhouette ↑":      round(sil, 4),
        "Davies-Bouldin ↓":  round(db,  4),
        "Ruido (%)":         round((~mask).mean() * 100, 1),
    }
    if y_true is not None:
        yt = np.array(y_true)[mask]
        resultado["ACC ↑"] = round(clustering_accuracy(yt, l_c), 4)
        resultado["NMI ↑"] = round(normalized_mutual_info_score(yt, l_c), 4)
        resultado["ARI ↑"] = round(adjusted_rand_score(yt, l_c), 4)

    return resultado


# ── Algoritmos ───────────────────────────────────────────────

def run_kmeans(X_train, X_test, k, n_init=15, seed=42):
    """Ajusta K-means en train y predice en test."""
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed, max_iter=300)
    labels_train = km.fit_predict(X_train)
    labels_test  = km.predict(X_test)
    return km, labels_train, labels_test


def run_dbscan(X, eps=0.5, min_samples=10):
    """Ajusta DBSCAN. No tiene predict estándar."""
    db     = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X)
    return db, labels


def run_pca(X_train, X_test, n_components=50, seed=42):
    """PCA: fit en train, transform en ambos."""
    pca  = PCA(n_components=n_components, random_state=seed)
    Xt   = pca.fit_transform(X_train)
    Xv   = pca.transform(X_test)
    return pca, Xt, Xv


def run_umap(X_train, X_test, n_components=10, n_neighbors=15, min_dist=0.1, seed=42):
    """UMAP: fit en train, transform en test."""
    import umap as _umap
    reducer = _umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        verbose=False,
    )
    Xt = reducer.fit_transform(X_train)
    Xv = reducer.transform(X_test)
    return reducer, Xt, Xv


def optimal_map(y_true, y_pred, n=10):
    """Mapeo óptimo cluster → clase real para análisis de error."""
    cost = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(p) < n:
            cost[int(p), int(t)] += 1
    row, col = linear_sum_assignment(cost.max() - cost)
    mapping  = {r: c for r, c in zip(row, col)}
    return np.array([mapping.get(int(p), -1) for p in y_pred])
