"""
src/data_loader.py
==================
Descarga, carga y trazabilidad del dataset MNIST.
CRISP-DM: Data Understanding
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def load_config(config_path="config/params.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_mnist(save_dir="data/raw"):
    """Descarga MNIST via Keras y lo persiste como .npz local (una sola vez)."""
    import tensorflow as tf

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    npz_path = save_dir / "mnist.npz"

    if npz_path.exists():
        print(f"[INFO] Dataset ya descargado. Cargando desde: {npz_path}")
        return _load_npz(npz_path)

    print("[INFO] Descargando MNIST via Keras (~11 MB) ...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    np.savez_compressed(
        npz_path,
        x_train=x_train, y_train=y_train,
        x_test=x_test,   y_test=y_test,
    )
    size_mb = npz_path.stat().st_size / 1e6
    print(f"[INFO] Guardado en: {npz_path}  ({size_mb:.1f} MB)")

    _save_metadata(x_train, x_test, npz_path)
    return (x_train, y_train), (x_test, y_test)


def load_mnist(data_dir="data/raw"):
    """Carga MNIST desde archivo local. Descarga automáticamente si no existe."""
    npz_path = Path(data_dir) / "mnist.npz"
    if not npz_path.exists():
        print("[INFO] Archivo local no encontrado. Iniciando descarga...")
        return download_mnist(data_dir)
    (x_train, y_train), (x_test, y_test) = _load_npz(npz_path)
    print(f"[INFO] MNIST cargado — train: {x_train.shape}  |  test: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


# ── helpers privados ─────────────────────────────────────────

def _load_npz(path):
    d = np.load(path)
    return (d["x_train"], d["y_train"]), (d["x_test"], d["y_test"])


def _save_metadata(x_train, x_test, npz_path):
    meta = {
        "dataset_name":  "MNIST",
        "authors":       "Yann LeCun, Corinna Cortes, Christopher Burges",
        "source_url":    "http://yann.lecun.com/exdb/mnist/",
        "keras_api":     "tensorflow.keras.datasets.mnist",
        "download_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_train":       int(len(x_train)),
        "n_test":        int(len(x_test)),
        "image_shape":   list(x_train.shape[1:]),
        "pixel_range":   [int(x_train.min()), int(x_train.max())],
        "n_classes":     10,
        "classes":       list(range(10)),
        "task":          "unsupervised clustering (labels used only for evaluation)",
        "local_path":    str(npz_path),
    }
    meta_path = Path("data/metadata.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Metadata guardada en: {meta_path}")
