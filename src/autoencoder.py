"""
src/autoencoder.py
==================
Autoencoder Convolucional (CAE) — arquitectura del paper AUEC.
CRISP-DM: Modeling — Técnica principal

Paper: Autoencoded UMAP-Enhanced Clustering for Unsupervised Learning
       Chavooshi & Mamonov, 2025 — arXiv:2501.07729

Nota sobre la loss:
  El paper usa J = λ·ψ(spectral_gap) + ρ·MSE  (loss compuesta).
  Este lab implementa solo MSE (reconstrucción) por viabilidad en CPU.
  La pérdida espectral requiere calcular eigenvalores del Laplaciano
  del grafo por batch, lo cual es intensivo computacionalmente sin GPU.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(latent_dim=10, input_shape=(28, 28, 1)):
    """
    Construye el Autoencoder Convolucional del paper AUEC.

    Arquitectura
    ------------
    Encoder:
      Input(28×28×1)
      → Conv2D(32, 3, relu, same) + BN + MaxPool(2)   →  14×14×32
      → Conv2D(64, 3, relu, same) + BN + MaxPool(2)   →   7×7×64
      → Flatten → Dense(256, relu) → Dense(latent_dim)  [espacio latente]

    Decoder:
      Dense(256) → Dense(3136) → Reshape(7, 7, 64)
      → Conv2DT(64, 3, relu, same) + UpSampling(2)    →  14×14×64
      → Conv2DT(32, 3, relu, same) + UpSampling(2)    →  28×28×32
      → Conv2DT(1,  3, sigmoid, same)                  →  28×28×1
    """
    # ── ENCODER ──────────────────────────────────────────────
    inputs = keras.Input(shape=input_shape, name="encoder_input")

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, padding="same")(x)          # 14×14×32

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, padding="same")(x)          # 7×7×64

    x = layers.Flatten()(x)                                # 3136
    x = layers.Dense(256, activation="relu")(x)
    latent = layers.Dense(latent_dim, name="latent")(x)    # espacio latente

    # ── DECODER ──────────────────────────────────────────────
    x = layers.Dense(256, activation="relu")(latent)
    x = layers.Dense(7 * 7 * 64, activation="relu")(x)
    x = layers.Reshape((7, 7, 64))(x)

    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)                          # 14×14×64

    x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)                          # 28×28×32

    decoded = layers.Conv2DTranspose(
        1, 3, activation="sigmoid", padding="same", name="decoded"
    )(x)                                                   # 28×28×1

    autoencoder = keras.Model(inputs, decoded, name="autoencoder")
    encoder     = keras.Model(inputs, latent,  name="encoder")
    return autoencoder, encoder


def train_autoencoder(autoencoder, x_train_cnn, cfg, validation_split=0.1):
    """Compila y entrena el autoencoder con MSE + callbacks."""
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(cfg["ae"]["learning_rate"]),
        loss="mse",
    )
    history = autoencoder.fit(
        x_train_cnn, x_train_cnn,
        epochs=cfg["ae"]["epochs"],
        batch_size=cfg["ae"]["batch_size"],
        validation_split=validation_split,
        shuffle=True,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg["ae"]["patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=3, verbose=0, min_lr=1e-5,
            ),
        ],
    )
    return history


def save_encoder_weights(encoder, output_dir="output/models"):
    """Guarda los pesos del encoder para uso en notebook de evaluación."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    encoder.save_weights(str(path / "encoder_weights.h5"))
    print(f"[INFO] Pesos del encoder guardados en: {path / 'encoder_weights.h5'}")


def load_encoder_weights(encoder, output_dir="output/models"):
    """Carga pesos del encoder previamente guardados."""
    path = Path(output_dir) / "encoder_weights.h5"
    if not path.exists():
        raise FileNotFoundError(
            f"[ERROR] No se encontraron pesos en: {path}\n"
            "Ejecuta primero el notebook 04_modeling.ipynb"
        )
    encoder.load_weights(str(path))
    print(f"[INFO] Pesos del encoder cargados desde: {path}")
    return encoder
