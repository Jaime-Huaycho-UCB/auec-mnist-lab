#!/bin/bash
# Ejecuta el lab AUEC en orden. Cada script es una fase CRISP-DM.

set -e
cd /workspace

echo ""
echo "=================================================="
echo "  Lab AUEC Clustering — Ejecucion automatica"
echo "=================================================="
echo ""

SCRIPTS=(
    "scripts/01_business_understanding.py"
    "scripts/02_data_understanding.py"
    "scripts/03_data_preparation.py"
    "scripts/04_modeling.py"
    "scripts/05_evaluation.py"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    echo "--------------------------------------------------"
    echo "  Ejecutando: $SCRIPT"
    echo "--------------------------------------------------"
    python "$SCRIPT"
    echo ""
done

echo "=================================================="
echo "  Ejecucion completa. Salidas en:"
echo "    output/figures/  -> graficas PNG"
echo "    output/results/  -> metricas_comparacion.csv"
echo "    output/models/   -> encoder_weights.h5"
echo "=================================================="
ls -lh output/figures/ output/results/ output/models/ 2>/dev/null || true
