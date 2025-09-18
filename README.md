

# 💸 Personal-Finance-App

This repository contains a web application for tracking personal expenses and budgeting. It features a Streamlit UI, FastAPI microservices for extraction and classification, and a SQLite database for storage. The app leverages machine learning models (transformers) for transaction classification and uses pdfplumber for PDF parsing. Docker and Docker Compose are used for reproducibility and deployment.

**Tech Stack:**

• **UI:** Streamlit  
• **End point:** FastAPI  
• **DB:** SQLite  
• **Reproducibility:** Docker (and Docker Compose)  
• **ML Libraries:** Trnasformer Library  
• **PDF parsing Library:** pdfplumber

---

**Repository structure**

```text
personal-finance-app/
├── .gitignore
├── README.md
├── requirements.txt
├── docker-compose.yml
├── backend/
│   ├── __init__.py
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── artifacts/
│   │   │   └── model_config.yaml
│   │   ├── config.py
│   │   ├── db.py
│   │   ├── main.py
│   │   ├── orchestrator.py
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── embedding.py
│   │       ├── llm.py
│   │       ├── zero_shot.py
│   ├── cli/
│   │   └── classify_csv.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── db.py
│   │   └── extraction.py
├── data/
│   └── trusted/
│       └── results.db
├── frontend/
│   ├── Dockerfile
│   ├── requirements.in
│   ├── requirements.txt
│   └── streamlit_app.py
├── services/
│   ├── classifier_api/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── requirements.in
│   │   ├── requirements.txt
│   │   └── schemas.py
│   └── extractor_api/
│       ├── Dockerfile
│       ├── main.py
│       ├── requirements.in
│       └── requirements.txt
```



---

## 🐳 Docker


1. After creating a `requirments.in` for each FastAPI (with only the needed dependencies) run the following to create the `requirments.txt` for such API:
    
    ```sh
    pip-compile services/extractor_api/requirements.in \
        -o services/extractor_api/requirements.txt
    ```


2. Building a docker image for each API:

    ```sh
    docker build -f services/extractor_api/Dockerfile -t personal-finance-app-extractor:dev .
    ```


3. We use a Bind mount instead of a named volume so that we can see the `.db` on the host (i.e. PC) instead of being embedded in Docker. So we run the image with the mounted bin volume:

    ```sh
    mkdir -p ./data
    docker run --rm -p 8000:8000 \
        -e DB_PATH=/data/results.db \
        -v "$(pwd)/data:/data" \
        personal-finance-app-extractor:dev
    ```



4. Create an internal network and run one image at a time after creating a network. We use a Bind mount instead of a Named volume so that we can see the `.db` on the host instead of being embedded in Docker. So we run each image that needs the db (the API) with the mounted bin volume. <br>Moreso, without creating an internal network we’d have to hard-code IP addresses (which change every time we restart, hence would not work). With the internal network the services are isolated from the rest of the host machine, but can discover each other by name reliably.

    ```sh
    docker network create pfa-net

    docker run -d --rm --name extractor \
        --network pfa-net \
        -p 8000:8000 \
        -e DB_PATH=/data/trusted/results.db \
        -v "$(pwd)/data:/data" \
        personal-finance-app-extractor:dev

    docker run -d --rm --name classifier \
        --network pfa-net \
        -p 8001:8001 \
        -e DB_PATH=/data/trusted/results.db \
        -e MODEL_CONFIG_YAML=/app/backend/classification/artifacts/model_config.yaml \
        -v "$(pwd)/data:/data" \
        -v "$(pwd)/hf_cache:/root/.cache/huggingface" \
        personal-finance-app-classifier:dev

    docker run --rm --name frontend \
        --network pfa-net \
        -p 8501:8501 \
        -e EXTRACTOR_API_URL=http://extractor:8000 \
        -e CLASSIFIER_API_URL=http://classifier:8001 \
        personal-finance-app-frontend:dev
    ```

5. Instead of point 4 we can orchestrate all such steps using docker-compose. Useful commands:

    ```sh
    docker compose build          # only once if containers not already built
    docker compose up -d       # start all
    docker compose ps          # see status
    docker compose logs -f     # tail logs
    docker compose down        # stop
    ```


ALternativelly for dev we can mount the code base as a volume so that updates are reflected in the containers.

---

## 🛠️ Useful commands

pip install packages:

When spinning up the app locally the URL is:  
`http://localhost:8501`

**Start venv:**
```sh
source .app_venv/bin/activate
```

**pip install packages:**
To safely bind packages to the python being used in the venv use:
```sh
python -m pip install package_name
```

**Run streamlit app:**
```sh
streamlit run frontend/streamlit_app.py
python -m streamlit run frontend/streamlit_app.py
```

**Stop streamlit app from terminal:**
```
control + c
```

**Remove from cache files that were tracked before adding them to gitignore:**
```sh
git rm --cached file_name
```

**Run backend with FastAPI:**
```sh
uvicorn services.extractor_api.main:app
```

**Swagger UI for API doc:**
`http://127.0.0.1:8000/docs`

**Start server with fast API:**
For production use:
```sh
fastapi run backend/api_main.py
```
For dev use:
```sh
python -m fastapi dev backend/api_main.py
```