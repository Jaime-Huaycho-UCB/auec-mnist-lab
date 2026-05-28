# Lab — Autoencoded UMAP-Enhanced Clustering (AUEC)

**Grupo:** JaimelA | **Materia:** Machine Learning — UCB Semestre 7
**Paper:** *Autoencoded UMAP-Enhanced Clustering for Unsupervised Learning*
Chavooshi & Mamonov, 2025 — [arXiv:2501.07729](https://arxiv.org/abs/2501.07729)

---

## Objetivo

Replicar y evaluar el pipeline AUEC sobre el dataset MNIST:

```
Imágenes (28×28) → Autoencoder Convolucional → Espacio latente (10D)
                 → UMAP (10D → 10D) → K-means / DBSCAN
```

Comparar contra tres baselines y demostrar que la combinación AE+UMAP
supera a cada técnica por separado en clustering no supervisado.

---

## Metodología CRISP-DM

| Fase | Script | Descripción |
|------|--------|-------------|
| Business Understanding | `scripts/01_business_understanding.py` | Objetivos, arquitectura del paper, hipótesis |
| Data Understanding     | `scripts/02_data_understanding.py`     | Carga de MNIST, EDA, distribuciones, varianza PCA |
| Data Preparation       | `scripts/03_data_preparation.py`       | Normalización, reshape, persistencia en `data/processed/` |
| Modeling               | `scripts/04_modeling.py`               | Baselines B1–B3, entrenamiento CAE, UMAP, K-means, DBSCAN |
| Evaluation             | `scripts/05_evaluation.py`             | Métricas, visualización 2D, matriz de confusión, conclusiones |

> **Orden de ejecución:** 01 → 02 → 03 → 04 → 05

---

## Arquitectura del proyecto

```
auec-mnist-lab/
├── config/
│   └── params.yaml          # Todos los hiperparámetros (única fuente de verdad)
├── data/
│   ├── raw/                 # mnist.npz (descargado automáticamente, ignorado en git)
│   └── processed/           # arrays .npy generados por script 03 (ignorado en git)
├── informe/                 # Informe escrito (elaborado por el grupo)
├── notebooks/
│   └── demo.ipynb           # Solo para presentación — carga outputs ya generados
├── output/                  # Generado al correr (ignorado en git)
│   ├── figures/             # Gráficas PNG
│   ├── models/              # Pesos del encoder (.h5)
│   └── results/             # metricas_comparacion.csv
├── scripts/                 # Lógica del lab — un archivo por fase CRISP-DM
│   ├── 01_business_understanding.py
│   ├── 02_data_understanding.py
│   ├── 03_data_preparation.py
│   ├── 04_modeling.py
│   └── 05_evaluation.py
├── src/                     # Módulos reutilizables
│   ├── data_loader.py       # Descarga y carga de MNIST
│   ├── preprocessing.py     # Normalización, reshape, persistencia
│   ├── autoencoder.py       # Arquitectura CAE + entrenamiento
│   ├── clustering.py        # Algoritmos + métricas de clustering
│   └── utils.py             # Estilos, guardado de figuras, reportes
├── Dockerfile
├── docker-compose.yml
├── run_lab.sh               # Ejecuta los 5 scripts en orden
└── requirements.txt
```

---

## Ejecución con Docker (recomendado)

```bash
# 1. Construir la imagen (solo la primera vez)
docker compose build

# 2. Correr el lab completo — ejecuta los 5 scripts en orden automáticamente
docker compose up

# 3. Los resultados aparecen en ./output/ de tu máquina local
docker compose down   # cuando termines
```

> El dataset MNIST (~11 MB) se descarga automáticamente en el primer run.
> Los resultados en `output/` son visibles en tu máquina gracias al volume mount.

**Nota sobre RAM:** El script 04 es el más pesado (UMAP + entrenamiento del AE).
Si Docker se queda sin memoria, reducir `n_train` de `15000` a `8000` en `config/params.yaml`.

---

## Ejecución local (sin Docker)

```bash
# Activar el entorno virtual
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# Instalar dependencias (si no están)
pip install -r requirements.txt

# Opción A — correr todo en secuencia
bash run_lab.sh

# Opción B — correr script por script
python scripts/01_business_understanding.py
python scripts/02_data_understanding.py
python scripts/03_data_preparation.py
python scripts/04_modeling.py
python scripts/05_evaluation.py
```

---

## Demo interactivo (después de correr los scripts)

```bash
source .venv/bin/activate
jupyter notebook notebooks/demo.ipynb
```

El `demo.ipynb` carga las figuras y métricas ya generadas para mostrarlas
de forma interactiva — útil para la presentación oral.

---

## Resultados de referencia

| Modelo | ACC ↑ | Silhouette ↑ | Davies-Bouldin ↓ |
|--------|-------|--------------|------------------|
| B1: K-means raw (784D)   | 52.3 % | 0.060 | 2.829 |
| B2: PCA(50) + K-means    | 52.2 % | 0.084 | 2.476 |
| B3: UMAP(raw) + K-means  | 77.5 % | 0.584 | 0.608 |
| **AUEC: AE+UMAP+KMeans** | **90.3 %** | **0.617** | **0.585** |
| AUEC: AE+UMAP+DBSCAN     | ~60 %  | — | — |

---

## Hiperparámetros

Todos centralizados en `config/params.yaml`. Para cambiar el experimento,
editar ese archivo y re-ejecutar los scripts desde el 03 en adelante.

---

## Dependencias principales

| Librería | Versión mínima | Uso |
|----------|----------------|-----|
| TensorFlow/Keras | 2.12 | Autoencoder convolucional |
| umap-learn | 0.5 | Reducción de dimensionalidad |
| scikit-learn | 1.3 | K-means, DBSCAN, PCA, métricas |
| scipy | 1.10 | Asignación húngara (ACC) |
| matplotlib / seaborn | 3.7 / 0.12 | Visualizaciones |
| pandas | 2.0 | Tablas de métricas |
