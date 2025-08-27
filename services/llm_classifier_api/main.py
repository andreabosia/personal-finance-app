# main.py
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import io
import pandas as pd
import json

from backend.llm_classifier.classifier import LLMConfig, classify_dataframe

app = FastAPI(title="LLM Transaction Classifier", version="1.0.0")


# -------- JSON endpoint --------
class Record(BaseModel):
    descrizione: str
    # You can add other fields if you want them echoed back
    # e.g., transaction_id: Optional[str] = None

class ClassifyJSONRequest(BaseModel):
    categories: List[str]
    records: List[Record]
    backend: Optional[str] = "ollama"            # "ollama" or "hf"
    model_name: Optional[str] = "llama3.1:8b"    # ollama tag OR HF model id
    temperature: Optional[float] = 0.0
    max_new_tokens: Optional[int] = 64


@app.post("/classify_records")
def classify_records(req: ClassifyJSONRequest):
    if not req.categories:
        return JSONResponse(status_code=400, content={"error": "categories cannot be empty"})

    df = pd.DataFrame([r.dict() for r in req.records])
    cfg = LLMConfig(
        backend=req.backend, model_name=req.model_name,
        temperature=req.temperature, max_new_tokens=req.max_new_tokens
    )

    out_df = classify_dataframe(df, categories=req.categories, llm_cfg=cfg)
    return {
        "categories": req.categories,
        "results": out_df.to_dict(orient="records"),
    }


# -------- CSV endpoint (multipart) --------
# Usage: form-data with:
# - file: CSV file with at least a 'descrizione' column
# - categories: JSON-encoded list of strings, e.g. '["groceries","restaurants","transport"]'
# - backend/model/temperature (optional) as simple fields

@app.post("/classify_csv")
async def classify_csv(
    file: UploadFile = File(...),
    categories: str = Form(...),               # JSON string list
    backend: Optional[str] = Form("ollama"),
    model_name: Optional[str] = Form("llama3.1:8b"),
    temperature: Optional[float] = Form(0.0),
    max_new_tokens: Optional[int] = Form(64),
):
    try:
        cats = json.loads(categories)
        if not isinstance(cats, list) or not all(isinstance(x, str) for x in cats):
            raise ValueError
    except Exception:
        return JSONResponse(status_code=400, content={"error": "categories must be a JSON list of strings"})

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"failed to read CSV: {e}"})

    cfg = LLMConfig(
        backend=backend, model_name=model_name,
        temperature=float(temperature), max_new_tokens=int(max_new_tokens)
    )
    out_df = classify_dataframe(df, categories=cats, llm_cfg=cfg)

    # Return CSV
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{file.filename.rsplit(".",1)[0]}_classified.csv"'},
    )