FROM python:3.11-slim
WORKDIR /app

COPY services/classifier_api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY backend/__init__.py /app/backend/__init__.py
COPY backend/ingestion /app/backend/ingestion
COPY backend/classification /app/backend/classification
COPY services/classifier_api /app/services/classifier_api

RUN mkdir -p /data
EXPOSE 8001
CMD ["uvicorn", "services.classifier_api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]