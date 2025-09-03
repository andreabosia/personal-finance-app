from fastapi import FastAPI, Query
from backend.classification.main import orchestrate_db, load_model_from_yaml
from fastapi.responses import JSONResponse
from backend.ingestion import db as dal
import os

app = FastAPI(title="Classifier API")

@app.post("/classify_transactions")
def classify_transactions(
    model_config_path: str = Query(
        os.environ.get(
            "MODEL_CONFIG_YAML",
            "/Users/andreabosia/Projects/personal-finance-app/backend/classification/artifacts/model_config.yaml",
        ),
        description="Path to YAML with model configs",
    ),
    merchant_col: str = Query("descrizione"),
    batch_size: int = Query(512, ge=1, le=5000),
):
    
    dal.init_db()
    summary = orchestrate_db(model_config_path, merchant_col=merchant_col, batch_size=batch_size)
    # example: [{"model_signature": "abc123", "predicted_rows": 240}, ...]
    return {"status": "ok", "summary": summary}

MODEL_YAML = os.getenv("MODEL_CONFIG_YAML",
    "backend/classification/artifacts/model_config.yaml"
)

@app.get("/models")
def list_models(model_config_path: str = Query(MODEL_YAML)):
    models = load_model_from_yaml(model_config_path)
    out = [{"name": getattr(m, "name", m.__class__.__name__),
            "model_signature": getattr(m, "signature", m.__class__.__name__)} for m in models]
    return {"models": out}

@app.get("/export_df")
def export_df(model_signature: str):
    df = dal.join_with_predictions(model_signature)
    # return full rows as JSON (Streamlit will convert to DataFrame)
    return JSONResponse({"rows": df.to_dict(orient="records")})