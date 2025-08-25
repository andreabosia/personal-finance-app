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
    uvicorn backend.api_main:app 

Swagger UI for API doc:
    http://127.0.0.1:8000/docs 

start server with fast API:
    for production use -> fastapi run backend/api_main.py
    for dev use -> python -m fastapi dev backend/api_main.py
    preffered way (to understand) -> uvicorn backend.api_main:app --reload




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


