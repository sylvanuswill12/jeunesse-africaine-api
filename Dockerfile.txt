FROM python:3.11-slim

# Dépendances système nécessaires à opencv et onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY poster_template.png .

# Télécharge le modèle U^2-Net au moment du build pour un premier démarrage rapide
RUN python -c "from rembg import new_session; new_session('u2net')"

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
