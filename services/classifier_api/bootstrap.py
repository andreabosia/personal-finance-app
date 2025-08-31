import pandas as pd
import os
import datetime as dt
from typing import Any, Dict, List, Tuple
from backend.classification.config import settings
from backend.classification.registry import StrategyRegistry
from backend.classification.orchestrator import Orchestrator
from backend.classification.store import ResultsStore
from backend.classification.models.utils import ClassificationRequest, ClassScore
from backend.classification.models.zero_shot import ZeroShotClassifier, ZeroShotConfig
from backend.classification.models.embedding import EmbeddingAnchorClassifier, EmbeddingAnchorConfig
from backend.classification.models.llm import LLMClassifier, LLMConfig



def build_registry() -> StrategyRegistry:
    reg = StrategyRegistry()
    # register strategies (can register multiple variants)
    reg.register(EmbeddingAnchorClassifier(EmbeddingAnchorConfig(class_anchors=settings.ANCHORS)))
    reg.register(ZeroShotClassifier(ZeroShotConfig(candidate_labels=settings.CANDIDATE_LABELS, multi_label=False)))
    #reg.register(LLMClassifier(LLMConfig(candidate_labels=settings.CANDIDATE_LABELS)))
    return reg

REGISTRY = build_registry()
ORCH = Orchestrator(REGISTRY)
STORE = ResultsStore(settings.RESULTS_DB)

def classify_csv_idempotent(csv_path: str, models: List[str], id_col: str, merchant_col: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    for col in [id_col, merchant_col]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}'")
    tx_ids = df[id_col].astype(str).tolist()

    # Build pairs
    strategies = [REGISTRY.get(m) for m in models]
    pairs: List[Tuple[str,str]] = [(s.name, s.signature) for s in strategies]
    existing = STORE.existing_keys(tx_ids, pairs)

    to_insert: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        tx_id = str(row[id_col])
        merchant = str(row[merchant_col]) if pd.notna(row[merchant_col]) else ""
        req = ClassificationRequest(text=merchant)
        for s in strategies:
            key = (tx_id, s.name, s.signature)
            if key in existing:
                continue
            res = s.predict_one(req)
            top = res.classes[0] if res.classes else ClassScore("UNKNOWN", 0.0)
            to_insert.append({
                "transaction_id": tx_id,
                "strategy": s.name,
                "signature": s.signature,
                "label": top.label,
                "score": float(top.score),
                "ts": dt.datetime.now(dt.UTC).isoformat(),
                "raw_json": None if res.raw is None else json_dumps_safe(res.raw),
            })
    STORE.upsert_rows(to_insert)
    rows = STORE.fetch_results(transaction_ids=tx_ids)
    return {"classified_new_rows": len(to_insert),
            "strategies": [{"name": s.name, "signature": s.signature} for s in strategies],
            "results": rows}

def json_dumps_safe(obj) -> str:
    import json
    try:
        return json.dumps(obj)
    except TypeError:
        return json.dumps(str(obj))
    


def _choose_best_labels(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """from multi-strategy rows -> best (max score) per transaction_id"""
    if not rows:
        return pd.DataFrame(columns=["transaction_id", "category", "score"])
    df = pd.DataFrame(rows)
    if "score" not in df.columns:
        df["score"] = 0.0
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    # keep highest score per transaction_id
    best = (
        df.sort_values(["transaction_id", "score"], ascending=[True, False])
          .drop_duplicates(subset=["transaction_id"], keep="first")
          .rename(columns={"label": "category"})[["transaction_id", "category", "score"]]
    )
    # normalize category
    if "category" in best.columns:
        best["category"] = best["category"].fillna("none").astype(str).str.strip().str.lower()
    return best

def _default_out_path(csv_path: str) -> str:
    base, ext = os.path.splitext(csv_path)
    return f"{base}_enriched{ext or '.csv'}"

def join_results_to_csv(csv_path: str, id_col: str, out_path: str | None = None) -> Dict[str, Any]:
    # robust read (keeps your original simple read; swap for a robust reader if you want)
    df = pd.read_csv(csv_path)

    if id_col not in df.columns:
        # light fallback: look for common alternatives
        for alt in ("transaction_id", "id", "txn_id", "tx_id", "ID"):
            if alt in df.columns:
                id_col = alt
                break
    if id_col not in df.columns:
        raise ValueError(f"CSV missing required column '{id_col}'")

    # ensure string ids
    df[id_col] = df[id_col].astype(str)
    tx_ids = df[id_col].tolist()

    # fetch classifications from SQLite and pick best label per tx
    rows = STORE.fetch_results(transaction_ids=tx_ids)
    best = _choose_best_labels(rows)
    if best.empty:
        # still write a passthrough enriched file so Streamlit can proceed
        out_path = out_path or _default_out_path(csv_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df_out = df.copy()
        if "category" not in df_out.columns:
            df_out["category"] = "none"
        df_out.to_csv(out_path, index=False, encoding="utf-8")
        return {
            "ok": True,
            "enriched_path": out_path,
            "tx_in_csv": len(df),
            "tx_with_classification": 0,
            "new_categories": 0,
        }

    # merge
    best = best.rename(columns={"transaction_id": id_col})
    df_enriched = df.merge(best, on=id_col, how="left")

    # prefer newly merged `category` if not present; if present, fill missing
    if "category_x" in df_enriched.columns and "category_y" in df_enriched.columns:
        df_enriched["category"] = df_enriched["category_x"].fillna(df_enriched["category_y"])
        df_enriched = df_enriched.drop(columns=["category_x", "category_y"])
    elif "category" not in df_enriched.columns:
        df_enriched["category"] = "none"

    # counts
    tx_with_cls = best[id_col].nunique()
    new_cats = df_enriched["category"].nunique()

    # write enriched
    out_path = out_path or _default_out_path(csv_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df_enriched.to_csv(out_path, index=False, encoding="utf-8")

    return {
        "ok": True,
        "enriched_path": out_path,
        "tx_in_csv": len(df),
        "tx_with_classification": int(tx_with_cls),
        "new_categories": int(new_cats),
    }