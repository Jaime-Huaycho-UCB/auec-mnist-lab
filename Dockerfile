FROM python:3.11-slim

LABEL maintainer="huaychojaime@gmail.com"
LABEL description="AUEC Clustering Lab — Defensa Papers ML UCB 2026"

# ── Sistema ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Directorio de trabajo ─────────────────────────────────────
WORKDIR /workspace

# ── Dependencias Python ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Código fuente ─────────────────────────────────────────────
# (los volumes de docker-compose sobreescriben esto en dev)
COPY src/        ./src/
COPY config/     ./config/
COPY notebooks/  ./notebooks/

# ── Variables de entorno ──────────────────────────────────────
ENV PYTHONPATH=/workspace
ENV JUPYTER_ENABLE_LAB=yes

# ── Carpetas de salida (creadas aquí; el volume las monta en host)
RUN mkdir -p output/figures output/models output/results \
             data/raw data/processed

EXPOSE 8888

CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''", \
     "--notebook-dir=/workspace/notebooks"]
