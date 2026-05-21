FROM python:3.11-slim

WORKDIR /app

# System deps for torch + lightweight image
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first to leverage Docker layer caching
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code. NOTE: the trained model is NOT baked into the
# image, it is pulled from the MLflow Model Registry at container start.
# This is the key separation-of-concerns improvement called out in the
# Padlet (training-serving skew, John et al. 2021).
COPY app/ /app/app/
COPY training/preprocess.py /app/training/preprocess.py
COPY training/__init__.py /app/training/__init__.py
COPY explainability/ /app/explainability/

# Make modules importable
ENV PYTHONPATH=/app

EXPOSE 5000

# Liveness probe target
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "-m", "flask", "--app", "app.app", "run", "--host", "0.0.0.0", "--port", "5000"]
