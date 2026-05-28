# Lab — Autoencoded UMAP-Enhanced Clustering (AUEC)

**Grupo:** JaimelA | **Materia:** Machine Learning — UCB Semestre 7  
**Paper:** *Autoencoded UMAP-Enhanced Clustering for Unsupervised Learning*  
Chavooshi & Mamonov, 2025 — [arXiv:2501.07729](https://arxiv.org/abs/2501.07729)

---

## Objetivo

Replicar y evaluar el pipeline AUEC sobre el dataset MNIST:

```
Imágenes (28×28) → Autoencoder Convolucional → Espacio latente (10D)
                 → UMAP (10D → 10D)  → K-means / DBSCAN
```

Comparar contra tres baselines y demostrar que la combinación AE+UMAP
supera a cada técnica por separado en clustering no supervisado.

---

## Metodología CRISP-DM

| Fase | Notebook | Descripción |
|------|----------|-------------|
| Business Understanding | `01_business_understanding.ipynb` | Objetivos del lab, descripción del paper AUEC, hipótesis |
| Data Understanding     | `02_data_understanding.ipynb`     | Carga de MNIST, EDA, distribuciones, visualización de muestras |
| Data Preparation       | `03_data_preparation.ipynb`       | Normalización, reshape, persistencia de arrays en `data/processed/` |
| Modeling               | `04_modeling.ipynb`               | Baselines B1–B3, entrenamiento del CAE, UMAP, K-means, DBSCAN |
| Evaluation             | `05_evaluation.ipynb`             | Métricas comparativas, visualización 2D, matriz de confusión, conclusiones |

> **Orden de ejecución:** 01 → 02 → 03 → 04 → 05

---

## Arquitectura del proyecto

```
proyecto/
├── config/
│   └── params.yaml          # Todos los hiperparámetros (única fuente de verdad)
├── data/
│   ├── raw/                 # mnist.npz (descargado automáticamente)
│   └── processed/           # arrays .npy generados por notebook 03
├── informe/                 # Informe escrito (elaborado por el estudiante)
├── notebooks/               # 5 notebooks CRISP-DM
├── output/
│   ├── figures/             # Gráficas PNG exportadas
│   ├── models/              # Pesos del encoder (.h5)
│   └── results/             # metricas_comparacion.csv
├── src/
│   ├── data_loader.py       # Descarga y carga de MNIST
│   ├── preprocessing.py     # Normalización, reshape, persistencia
│   ├── autoencoder.py       # Arquitectura CAE + entrenamiento
│   ├── clustering.py        # Algoritmos + métricas de clustering
│   └── utils.py             # Estilos, guardado de figuras, reportes
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Ejecución con Docker (recomendado)

```bash
# 1. Construir la imagen
docker compose build

# 2. Levantar el servidor Jupyter
docker compose up

# 3. Abrir en el navegador
#    http://localhost:8888

# 4. Ejecutar los notebooks en orden: 01 → 02 → 03 → 04 → 05

# 5. Los resultados quedan en ./output/ (montado como volume)
docker compose down   # cuando termines
```

> **Nota:** La carpeta `output/` en tu sistema local se actualiza en tiempo real
> gracias al volume mount. No necesitas copiar archivos desde el contenedor.

---

## Ejecución local (sin Docker)

```bash
# Activar el entorno virtual
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Instalar dependencias (si no están)
pip install -r requirements.txt

# Iniciar Jupyter
jupyter notebook notebooks/
```

---

## Resultados principales

| Modelo | ACC ↑ | NMI ↑ | ARI ↑ | Silhouette ↑ | Davies-Bouldin ↓ |
|--------|-------|-------|-------|--------------|------------------|
| B1: K-means (raw 784D)   | 52.3 % | — | — | 0.060 | 2.829 |
| B2: PCA(50) + K-means    | 52.2 % | — | — | 0.084 | 2.476 |
| B3: UMAP(raw) + K-means  | 77.5 % | — | — | 0.584 | 0.608 |
| **AUEC: AE+UMAP+KMeans** | **90.3 %** | — | — | **0.617** | **0.585** |
| AUEC: AE+UMAP+DBSCAN     | 60.6 % | — | — | — | — |

*Los valores de NMI y ARI se calculan en el notebook 05.*

---

## Configuración de hiperparámetros

Todos los hiperparámetros se centralizan en `config/params.yaml`.
Para cambiar el experimento, edita ese archivo y re-ejecuta los notebooks.

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
