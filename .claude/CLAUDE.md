# Lab AUEC Clustering — Contexto para Claude
**Grupo:** JaimelA | **Materia:** Machine Learning, UCB Semestre 7
**Paper:** *Autoencoded UMAP-Enhanced Clustering for Unsupervised Learning*
Chavooshi & Mamonov, 2025 — arXiv:2501.07729

---

## Qué es este proyecto

Es el laboratorio reproducible que acompaña la defensa oral del paper AUEC.
El lab replica el pipeline del paper sobre MNIST y sirve como evidencia técnica
para la defensa (vale como parte del 3er parcial de ML).

**El lab NO es el informe ni la presentación** — esos los hace el grupo por su cuenta.
Lo que hay aquí es únicamente el código, los notebooks y la infraestructura Docker.

---

## Estado actual

### Completado
- [x] 5 notebooks CRISP-DM (`notebooks/01` al `05`)
- [x] 6 módulos Python en `src/`
- [x] `config/params.yaml` — hiperparámetros centralizados
- [x] Docker + docker-compose con ejecución automática via `run_lab.sh`
- [x] `.gitignore` — ignora resultados generados (output/, data/)
- [x] Git inicializado con credenciales de Jaime Huaycho

### Pendiente (para el compañero que continúe)
- [ ] Ejecutar el lab completo y verificar que todos los notebooks corren sin errores
  - El notebook 04 puede dar OOM en Docker con RAM limitada
  - Solución rápida: reducir `n_train` de 15000 a 8000 en `config/params.yaml`
- [ ] Informe escrito (carpeta `informe/`) — lo hace el grupo, no Claude
- [ ] Presentación en PowerPoint/Google Slides — la hace el grupo

---

## Cómo correr el lab

```bash
# Con Docker (recomendado)
docker compose build   # primera vez solamente
docker compose up      # corre todos los notebooks automáticamente

# Sin Docker
source .venv/bin/activate
jupyter notebook notebooks/
# Ejecutar en orden: 01 → 02 → 03 → 04 → 05
```

Los resultados aparecen en `output/` de tu máquina local (volume mount).

---

## Arquitectura del código

```
auec-mnist-lab/
├── config/params.yaml      ← ÚNICA fuente de verdad para hiperparámetros
├── scripts/                ← 5 fases CRISP-DM, ejecutar en orden con python
│   ├── 01_business_understanding.py   (objetivos, arquitectura del paper)
│   ├── 02_data_understanding.py       (EDA de MNIST)
│   ├── 03_data_preparation.py         (normalización, reshape, persistencia)
│   ├── 04_modeling.py                 (baselines B1/B2/B3 + AUEC)
│   └── 05_evaluation.py              (métricas, viz, análisis de error)
├── notebooks/
│   └── demo.ipynb          ← solo para presentación/demo, carga outputs ya generados
├── src/
│   ├── data_loader.py      ← descarga/carga MNIST, load_config()
│   ├── preprocessing.py    ← normalize, reshape_cnn/flat, save/load_processed
│   ├── autoencoder.py      ← build_autoencoder, train_autoencoder, save/load weights
│   ├── clustering.py       ← run_kmeans/dbscan/pca/umap, evaluar_clustering
│   └── utils.py            ← estilos, save_fig, print_hallazgo, COLORS
├── data/                   ← generado al correr (ignorado en git)
│   ├── raw/mnist.npz       ← descargado automáticamente en nb 02
│   └── processed/*.npy     ← generados por nb 03
├── output/                 ← generado al correr (ignorado en git)
│   ├── figures/            ← PNGs guardados por save_fig()
│   ├── results/            ← metricas_comparacion.csv
│   └── models/             ← encoder_weights.h5
├── Dockerfile
├── docker-compose.yml
├── run_lab.sh              ← ejecuta nbconvert --execute en orden
└── requirements.txt
```

### Flujo de datos entre notebooks

```
nb 03 guarda → data/processed/*.npy
nb 04 lee  ← data/processed/*.npy
nb 04 guarda → data/processed/ltr_*.npy, z_train*.npy
             → output/models/encoder_weights.h5
nb 05 lee  ← data/processed/ltr_*.npy, z_train*.npy
nb 05 guarda → output/results/metricas_comparacion.csv
```

---

## Decisiones técnicas importantes

| Decisión | Por qué |
|----------|---------|
| MSE en lugar de loss espectral | La loss del paper requiere calcular eigenvalores del Laplaciano del grafo por batch — inviable en CPU. Se documenta como limitación. |
| subset 15 000 / 3 000 | Viabilidad en CPU. El paper usa 60 000/10 000. Los resultados siguen siendo representativos. |
| seed=42 en todo | Reproducibilidad. Fijado en params.yaml, pasado a todas las funciones. |
| UMAP 10D para clustering, 2D para visualización | Separación de concerns: 10D conserva más información para K-means, 2D es solo para graficar. |
| save/load_processed en .npy | Permite que cada notebook sea independiente y no recalcule todo desde cero. |

---

## Resultados de referencia (ejecución anterior)

Estos son los valores que se obtuvieron en una ejecución previa. Sirven para
verificar que el lab funciona correctamente:

| Modelo | ACC | Silhouette | Davies-Bouldin |
|--------|-----|------------|----------------|
| B1: KMeans raw (784D) | ~52% | ~0.060 | ~2.83 |
| B2: PCA(50)+KMeans | ~52% | ~0.084 | ~2.48 |
| B3: UMAP(raw)+KMeans | ~77% | ~0.584 | ~0.61 |
| AUEC: AE+UMAP+KMeans | ~90% | ~0.617 | ~0.58 |

Si los resultados difieren mucho de estos valores, revisar:
1. Que `seed=42` está en params.yaml
2. Que se ejecutaron los notebooks en orden (03 antes de 04, 04 antes de 05)
3. Que no hubo OOM en el notebook 04 (reducir n_train si es necesario)

---

## Qué NO tocar sin razón

- `src/clustering.py` → la función `clustering_accuracy` usa Hungarian algorithm,
  no cambiar la lógica sin entender bien scipy.optimize.linear_sum_assignment
- `src/autoencoder.py` → la arquitectura del CAE replica exactamente el paper,
  cambiarla invalida la comparación con el paper original
- `config/params.yaml` → es la única fuente de verdad; si cambias un valor,
  tienes que re-ejecutar todos los notebooks desde el 03

---

## Convenciones de código

- Idioma: español para comentarios/prints, inglés para nombres de funciones/variables
- Funciones en `src/` son puras (sin side effects globales)
- Toda figura se guarda con `save_fig(nombre)` de `src/utils.py`
- Hallazgos se imprimen con `print_hallazgo(tag, msgs)` — no usar print directo
- Rutas siempre desde la raíz del proyecto (notebooks hacen `os.chdir('..')`)

---

## Contexto académico

- **Materia:** Machine Learning, Universidad Católica Boliviana (UCB)
- **Semestre:** 7, año 2026
- **Evaluación:** Defensa de papers — 3er parcial
- **Rúbrica relevante para el lab (20 pts):** Laboratorio reproducible con baseline + técnica principal
- **Rúbrica relevante para métricas (15 pts):** Uso adecuado de métricas y análisis de error
- **Rúbrica crítica/ética (15 pts):** Limitaciones, sesgo, ética y generalización → cubierto en nb 05 sección 5.6
- **El lab NO reemplaza el informe ni la presentación** — son entregables separados
