# personal-finance-app
Web App that allows to track personal expenses and budgeting. Developed leveraging streamline for UI, FastAPI  for data processing, Docker for containerisation, airflow for automation. 



start venv:
    source .app_venv/bin/activate

pip install packages:
    to safely bind packages to the python being used in the venv use python -m pip install package_name instead of just pip install package_name
    NOTE: square brackets are interpreted as a globbing pattern in zsh --> python -m pip install "camelot-py[cv]"

run streamlit app:
   streamlit run frontend/streamlit_app.py
   python -m streamlit run frontend/streamlit_app.py

stop streamlit app from terminal:
    control + c

remove from cache files that were tracked before adding them to gitignore:
    git rm --cached file_name

run backend with FastAPI:
    uvicorn services.extractor_api.main:app 

Swagger UI for API doc:
    http://127.0.0.1:8000/docs 

start server with fast API:
    for production use -> fastapi run backend/api_main.py
    for dev use -> python -m fastapi dev backend/api_main.py
    preffered way (to understand) -> uvicorn backend.api_main:app --reload

##
DOCKERIZING

1) After creating a requirments.in for each fast API (with only the needed dependencies) run the following to create the requirments.txt for such api
    pip-compile services/extractor_api/requirements.in \
        -o services/extractor_api/requirements.txt

2) Building a docker image for each API:

    “Build a Docker image using the file services/extractor_api/Dockerfile, name/tag it pfa-extractor:dev, and use the current repo as the build context so the COPY instructions can find the code.”

    docker build -f services/extractor_api/Dockerfile -t personal-finance-app-extractor:dev .

3) we use a Bind mount instead of a named volume so that we can see th .db on the host (my pc) instead of being embedded in Docker. So we run the image with the mounted bin volume.
    mkdir -p ./data
    docker run --rm -p 8000:8000 \
    -e DB_PATH=/data/results.db \
    -v "$(pwd)/data:/data" \
    personal-finance-app-extractor:dev

3.1)
    run one image at a time after creating a network

    NOTE: Without it You’d have to hard-code IP addresses (which change every time you restart), or expose all ports to your host and use host.docker.internal.With it Your services are isolated from the rest of your machine, but can discover each other by name reliably.

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

    open the app from browser --> http://localhost:8501

4) instead of point 3.1 orchestrate using docker-compose
    docker compose up -d       # start all
    docker compose ps          # see status
    docker compose logs -f     # tail logs

    docker compose down            # stop
    docker compose down -v         # stop + remove named volumes (not needed here since we bind-mount ./data)    



Repository structure

your-project/
│── pyproject.toml           # poetry config (dependencies, metadata)
│── poetry.lock
│── docker-compose.yml       # orchestrates services locally (optional)
│── README.md

├── backend/                 # core business logic (independent of FastAPI/Streamlit)
│   ├── ingestion/
│   │   └── extraction.py    # PDF → tabular data
│   │
│   ├── preprocessing/
│   │   └── preprocessing.py # merchant → embedding → similarity features
│   │
│   ├── training/
│   │   ├── clustering_model.py  # TxClustering (BaseEstimator-style)
│   │   └── train.py             # orchestrates fit/eval/save_artifacts
│   │
│   ├── inference/
│   │   ├── predictor.py         # loads artifacts, exposes predict()
│   │   └── preprocessing.py     # lightweight reuse of preprocessing at inference
│   │
│   └── common/                  # shared utils (used by both training & inference)
│       ├── __init__.py
│       ├── schema.py            # Pydantic input/output schemas
│       ├── embeddings.py        # multilingual embedding loader + caching
│       ├── similarity.py        # cosine similarity, distance utils
│       ├── artifacts.py         # save/load model artifacts & metadata
│       ├── logging.py           # shared logger config
│       └── constants.py         # paths, env vars, model_version etc.

├── services/                # FastAPI microservices (entrypoints)
│   ├── extractor_api/
│   │   ├── main.py          # FastAPI app for /extract
│   │   └── Dockerfile
│   │
│   ├── training_api/
│   │   ├── main.py          # FastAPI app for /train
│   │   └── Dockerfile
│   │
│   └── inference_api/
│       ├── main.py          # FastAPI app for /predict
│       └── Dockerfile

├── artifacts/               # persisted models, embeddings, metadata
│   └── tx_clustering/
│       └── 1.0.0/
│           ├── pipeline.pkl
│           ├── category_embeddings.pkl
│           └── metadata.json

└── tests/                   # pytest unit & integration tests
    ├── test_ingestion.py
    ├── test_preprocessing.py
    ├── test_training.py
    ├── test_inference.py
    └── test_api.py
|
|
|    
└── frontend/                   # UI
    ├── streamlit_app.py


