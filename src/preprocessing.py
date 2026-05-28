"""
src/preprocessing.py
====================
Pipeline de preprocesamiento para MNIST.
CRISP-DM: Data Preparation
"""

from pathlib import Path

import numpy as np


def create_subsets(x_train, y_train, x_test, y_test,
                   n_train=15_000, n_test=3_000, seed=42):
    """Selecciona subset reproducible para viabilidad en CPU."""
    rng = np.random.default_rng(seed)
    idx_tr = rng.choice(len(x_train), min(n_train, len(x_train)), replace=False)
    idx_te = rng.choice(len(x_test),  min(n_test,  len(x_test)),  replace=False)
    return (
        x_train[idx_tr], y_train[idx_tr],
        x_test[idx_te],  y_test[idx_te],
    )


def normalize(x):
    """Normaliza imágenes uint8 al rango [0.0, 1.0]."""
    return x.astype("float32") / 255.0


def reshape_cnn(x):
    """Agrega dimensión de canal: (N, 28, 28) → (N, 28, 28, 1)."""
    return x[..., np.newaxis]


def reshape_flat(x):
    """Aplana imagen a vector: (N, 28, 28) → (N, 784)."""
    return x.reshape(len(x), -1)


def save_processed(arrays: dict, save_dir="data/processed"):
    """Guarda arrays preprocesados como .npy para uso entre notebooks."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        path = save_dir / f"{name}.npy"
        np.save(path, arr)
    print(f"[INFO] {len(arrays)} arrays guardados en: {save_dir}")


def load_processed(names: list, load_dir="data/processed"):
    """Carga arrays preprocesados desde .npy."""
    load_dir = Path(load_dir)
    arrays = {}
    for name in names:
        path = load_dir / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"[ERROR] No se encontró: {path}\n"
                "Ejecuta primero el notebook 03_data_preparation.ipynb"
            )
        arrays[name] = np.load(path, allow_pickle=False)
    print(f"[INFO] {len(arrays)} arrays cargados desde: {load_dir}")
    return arrays
