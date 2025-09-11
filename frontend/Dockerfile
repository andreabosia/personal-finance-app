FROM python:3.11-slim

WORKDIR /app

# 1) Install only FE deps
COPY frontend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 2) Copy just the frontend code (and anything it imports directly)
COPY frontend /app/frontend

# 3) Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]