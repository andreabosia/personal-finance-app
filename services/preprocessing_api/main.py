# services/preprocessing_api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd

from backend.preprocessing.preprocessing import MerchantCategoryPreprocessor, Paths

# ---- Config (adjust paths or load from YAML/env) ----
paths = Paths(
    merchant_emb_parquet="artifacts/features/merchant_embeddings.parquet",
    category_emb_parquet="artifacts/features/category_embeddings.parquet",
    sims_parquet="artifacts/features/tx_category_sims.parquet"
)
pp = MerchantCategoryPreprocessor(paths, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", top_k=5)

# ---- Schemas ----
class TxIn(BaseModel):
    transaction_id: str = Field(..., descrizione="Stable unique id for the transaction")
    descrizione: str = Field(..., descrizione="Raw or cleaned merchant descrizione")

class CategoryIn(BaseModel):
    category_id: str
    category_text: str
    language: Optional[str] = None

class EmbedMerchantsRequest(BaseModel):
    transactions: List[TxIn]

class EmbedCategoriesRequest(BaseModel):
    categories: List[CategoryIn]

class PreprocessRequest(BaseModel):
    transactions: List[TxIn]
    # Optional: supply categories again, otherwise reuse existing ones in parquet
    categories: Optional[List[CategoryIn]] = None
    compute_sims: bool = True
    top_k: Optional[int] = None

# ---- App ----
app = FastAPI(title="Preprocessing API", version="1.0.0")

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_name": pp.model_name}

@app.post("/embed-merchants")
def embed_merchants(req: EmbedMerchantsRequest):
    if not req.transactions:
        raise HTTPException(status_code=400, detail="transactions empty")
    df = pd.DataFrame([t.model_dump() for t in req.transactions])
    out = pp.embed_and_append_merchants(df, transaction_id_col="transaction_id", merchant_text_col="descrizione")
    return {
        "processed": len(df),
        "total_rows_in_store": int(len(out)),
        "model_name": pp.model_name
    }

@app.post("/embed-categories")
def embed_categories(req: EmbedCategoriesRequest):
    if not req.categories:
        raise HTTPException(status_code=400, detail="categories empty")
    df = pd.DataFrame([c.model_dump() for c in req.categories])
    out = pp.embed_and_append_categories(df, category_id_col="category_id", category_text_col="category_text", language_col="language")
    return {
        "processed": len(df),
        "total_rows_in_store": int(len(out)),
        "model_name": pp.model_name
    }

@app.post("/compute-similarities")
def compute_similarities(top_k: Optional[int] = None):
    sims = pp.compute_and_append_similarities(top_k=top_k)
    return {"total_rows_in_store": int(len(sims)), "model_name": pp.model_name}

@app.post("/preprocess")
def preprocess(req: PreprocessRequest):
    # 1) categories (optional each call; otherwise use stored)
    if req.categories:
        cats_df = pd.DataFrame([c.model_dump() for c in req.categories])
        pp.embed_and_append_categories(cats_df, category_id_col="category_id", category_text_col="category_text", language_col="language")
    # 2) merchants
    tx_df = pd.DataFrame([t.model_dump() for t in req.transactions])
    pp.embed_and_append_merchants(tx_df, transaction_id_col="transaction_id", merchant_text_col="descrizione")
    # 3) similarities
    if req.compute_sims:
        sims = pp.compute_and_append_similarities(top_k=req.top_k)
        return {"status": "ok", "sims_rows": int(len(sims)), "model_name": pp.model_name}
    return {"status": "ok", "message": "embeddings updated", "model_name": pp.model_name}