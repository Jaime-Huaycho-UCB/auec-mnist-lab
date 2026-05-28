"""
src/utils.py
============
Funciones auxiliares: estilo, visualización, guardado, reportes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Paleta de colores consistente ────────────────────────────
COLORS = {
    "primary":   "#2E74B5",
    "secondary": "#1F3864",
    "accent":    "#E06C00",
    "light":     "#BDD7EE",
    "good":      "#1a9850",
    "warn":      "#fc8d59",
    "bad":       "#d73027",
}


def set_plot_style():
    """Aplica estilo visual consistente a todas las figuras."""
    plt.rcParams.update({
        "figure.dpi":        100,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
    })


def save_fig(name, output_dir="output/figures"):
    """Guarda la figura activa como PNG en output_dir."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{name}.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    print(f"[INFO] Figura guardada: {path}")


def print_hallazgo(tag, msgs):
    """Imprime hallazgo formateado al estilo del proyecto ejemplo."""
    if isinstance(msgs, str):
        msgs = [msgs]
    print(f"\n  [{tag:14s}] {msgs[0]}")
    for m in msgs[1:]:
        print(f"  {'':16s} {m}")


def print_separador(titulo=""):
    line = "=" * 65
    print(f"\n{line}")
    if titulo:
        print(f"  {titulo}")
        print(line)


def resultados_a_csv(resultados: list, output_dir="output/results"):
    """Exporta tabla de métricas a CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df   = pd.DataFrame(resultados)
    path = Path(output_dir) / "metricas_comparacion.csv"
    df.to_csv(path, index=False)
    print(f"[INFO] Métricas exportadas a: {path}")
    return df


def cluster_purity_report(labels_train, y_train, k=10):
    """Calcula pureza por cluster y retorna DataFrame ordenado."""
    rows = []
    for c in range(k):
        mask = labels_train == c
        if mask.sum() == 0:
            continue
        vals, counts = np.unique(y_train[mask], return_counts=True)
        purity   = counts.max() / counts.sum()
        dominant = int(vals[counts.argmax()])
        rows.append({
            "Cluster":          c,
            "Pureza":           round(purity, 4),
            "Clase dominante":  dominant,
            "Tamaño":           int(mask.sum()),
        })
    return pd.DataFrame(rows).sort_values("Pureza").reset_index(drop=True)
