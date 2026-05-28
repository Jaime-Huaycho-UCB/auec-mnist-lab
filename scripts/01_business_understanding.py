"""
scripts/01_business_understanding.py
=====================================
CRISP-DM Fase 1 — Business Understanding
Verifica la configuracion del proyecto y presenta el contexto del paper AUEC.
"""

import sys
sys.path.insert(0, '.')

from src.data_loader import load_config
from src.utils import print_separador, print_hallazgo

cfg = load_config()

print_separador("Lab AUEC Clustering | Defensa Papers ML UCB 2026")
print("  Grupo JaimelA")
print("  Paper: Autoencoded UMAP-Enhanced Clustering for Unsupervised Learning")
print("  Chavooshi & Mamonov, 2025 — arXiv:2501.07729")

print_separador("Objetivos del lab")
print("  1. Replicar el pipeline AUEC: CAE + UMAP + K-means sobre MNIST")
print("  2. Comparar con baselines: B1 KMeans raw, B2 PCA+KMeans, B3 UMAP(raw)+KMeans")
print("  3. Evaluar con ACC, NMI, ARI, Silhouette, Davies-Bouldin")
print("  4. Identificar la contribucion de cada componente del pipeline")

print_separador("Pipeline AUEC")
print("  Input (28x28) --> Autoencoder Convolucional --> Z (10D latente)")
print("                                              --> UMAP (10D -> 10D)")
print("                                              --> K-means / DBSCAN")

print_separador("Hipotesis")
print("  H1: AUEC supera a los tres baselines en ACC y metricas internas")
print("  H2: UMAP(AE latente) produce mejores clusters que UMAP(raw 784D)")
print("  H3: El AE aprende representaciones mejores que PCA lineal")

print_separador("Configuracion cargada")
for section, vals in cfg.items():
    print(f"  {section}: {vals}")

print_separador("Relacion con la materia")
temas = [
    ("Clustering no supervisado", "K-means, DBSCAN, metricas internas/externas"),
    ("Reduccion de dimensionalidad", "PCA (lineal) vs UMAP (no lineal manifold)"),
    ("Deep Learning", "Autoencoder Convolucional como extractor de features"),
    ("Evaluacion de modelos", "ACC Hungarian, NMI, ARI, Silhouette, Davies-Bouldin"),
    ("CRISP-DM", "Metodologia que estructura todo el lab"),
]
for tema, aplicacion in temas:
    print(f"  {tema:35s} -> {aplicacion}")

print("\n  [OK] 01_business_understanding completado.\n")
