from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException
from backend.classification.main import (
    load_model_from_yaml,         
    load_models_from_yaml_bytes, 
    orchestrate_db_from_classifiers
)
from fastapi.responses import JSONResponse
from backend.ingestion import db as dal
import os, tempfile, shutil

app = FastAPI(title="Classifier API")

DEFAULT_MODEL_YAML = os.environ.get(
    "MODEL_CONFIG_YAML",
    "backend/classification/artifacts/model_config.yaml"
)

@app.post("/classify_transactions")
async def classify_transactions(
    model_config_file: UploadFile | None = File(None),
    merchant_col: str = Form("descrizione"),
    batch_size: int = Form(512),
):
    dal.init_db()

    try:
        if model_config_file is not None:
            # Read bytes directly
            yaml_bytes = await model_config_file.read()
            classifiers = load_models_from_yaml_bytes(yaml_bytes)
        else:
            # Fall back to default path inside this container.
            classifiers = load_model_from_yaml(DEFAULT_MODEL_YAML)

        summary = orchestrate_db_from_classifiers(
            classifiers, merchant_col=merchant_col, batch_size=batch_size
        )
        return {"status": "ok", "summary": summary}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"YAML not found: {DEFAULT_MODEL_YAML}")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Last-resort error surface
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
def list_models_from_db():
    try:
        df = dal.list_model_signatures()
        models = [
            {
                "model_signature": row["model_signature"],
                "n_predictions": int(row["n_predictions"] or 0),
                "last_predicted_at": row["last_predicted_at"],
            }
            for _, row in df.iterrows()
        ]
        return {"models": models}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/export_df")
def export_df(model_signature: str):
    df = dal.join_with_predictions(model_signature)
    # return full rows as JSON (Streamlit will convert to DataFrame)
    return JSONResponse({"rows": df.to_dict(orient="records")})



# --------------------------------------------------------------------------------------------------------------------
# Debug endpoints



@app.get("/debug/db_info")
def db_info():
    try:
        from backend.ingestion.db import DB_PATH, get_conn
        with get_conn() as conn:
            tx = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            pr = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            sigs = conn.execute("SELECT COUNT(DISTINCT model_signature) FROM predictions").fetchone()[0]
        return {
            "db_path": str(DB_PATH),
            "transactions": int(tx),
            "predictions": int(pr),
            "distinct_signatures": int(sigs),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/debug/predictions_head")
def predictions_head(limit: int = 10):
    try:
        from backend.ingestion.db import get_conn
        with get_conn() as conn:
            rows = conn.execute("""
                SELECT id, model_signature, label, score, predicted_at
                FROM predictions
                ORDER BY predicted_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        # convert sqlite row tuples to dicts
        cols = ["id","model_signature","label","score","predicted_at"]
        out = [dict(zip(cols, r)) for r in rows]
        return {"rows": out}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})