#!/bin/bash
# Ejecuta todos los notebooks CRISP-DM en orden y guarda las salidas.

set -e
cd /workspace

echo ""
echo "=================================================="
echo "  Lab AUEC Clustering — Ejecucion automatica"
echo "=================================================="
echo ""

NOTEBOOKS=(
    "notebooks/01_business_understanding.ipynb"
    "notebooks/02_data_understanding.ipynb"
    "notebooks/03_data_preparation.ipynb"
    "notebooks/04_modeling.ipynb"
    "notebooks/05_evaluation.ipynb"
)

for NB in "${NOTEBOOKS[@]}"; do
    echo "--------------------------------------------------"
    echo "  Ejecutando: $NB"
    echo "--------------------------------------------------"
    jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=1800 \
        --ExecutePreprocessor.kernel_name=python3 \
        "$NB"
    echo "  [OK] $NB completado."
    echo ""
done

echo "=================================================="
echo "  Ejecucion completa. Salidas guardadas en:"
echo "    output/figures/  -> graficas PNG"
echo "    output/results/  -> metricas_comparacion.csv"
echo "    output/models/   -> encoder_weights.h5"
echo "=================================================="
echo ""
ls -lh output/figures/ output/results/ output/models/ 2>/dev/null || true
