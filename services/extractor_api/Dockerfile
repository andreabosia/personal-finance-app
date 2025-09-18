FROM python:3.11-slim

WORKDIR /app

# 1) Install only classifier deps
COPY services/extractor_api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 2) Copy just the modules this service imports
# COPY backend/__init__.py /app/backend/__init__.py
# COPY backend/ingestion /app/backend/ingestion
# COPY backend/classification /app/backend/classification
# COPY services/classifier_api /app/services/classifier_api

# 3) Data dir for the SQLite DB (we'll mount /data from host)
RUN mkdir -p /data

EXPOSE 8001

# Keep a single worker (SQLite-friendly)
CMD ["uvicorn", "services.classifier_api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]