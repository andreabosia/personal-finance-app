from __future__ import annotations
import os
import yaml
import pandas as pd
from typing import List, Dict, Any
from backend.classification.models.embedding import (
    EmbeddingAnchorConfig, EmbeddingAnchorClassifier
)
from backend.ingestion import db as dal  # <- stdlib sqlite3 DAL

"""
Reads model config(s), loads classifiers, finds rows missing predictions for each
model in SQLite, computes predictions, and UPSERTs them to the `predictions` table.
"""

def load_model_from_yaml(yaml_path: str):
    classifiers_list = []
    with open(yaml_path, "r") as f:
        cfg_dict = yaml.safe_load(f) or {}

    for key, cfg in cfg_dict.items():
        if key.lower() == "embeddingconfig":
            model_config = EmbeddingAnchorConfig(**cfg)
            model = EmbeddingAnchorClassifier(model_config)
            classifiers_list.append(model)
        else:
            raise ValueError(f"Unsupported model type in YAML: {key}")
    return classifiers_list

def load_models_from_cfg_dict(cfg_dict: Dict[str, Any]) -> List[Any]:
    """
    Build classifiers from an already-parsed YAML dict.
    """
    classifiers: List[Any] = []
    for key, cfg in (cfg_dict or {}).items():
        if key.lower() == "embeddingconfig":
            model_config = EmbeddingAnchorConfig(**cfg)
            model = EmbeddingAnchorClassifier(model_config)
            classifiers.append(model)
        else:
            raise ValueError(f"Unsupported model type in YAML: {key}")
    return classifiers

def load_models_from_yaml_bytes(yaml_bytes: bytes) -> List[Any]:
    """
    Build classifiers directly from YAML bytes (no temp file).
    """
    cfg_dict = yaml.safe_load(yaml_bytes.decode("utf-8")) or {}
    return load_models_from_cfg_dict(cfg_dict)

def orchestrate_db_from_classifiers(
    classifiers: Iterable[Any],
    merchant_col: str = "descrizione",
    batch_size: int = 512,
) -> list[dict[str, Any]]:
    """
    For each classifier:
      - compute its model_signature
      - fetch rows missing predictions for that signature
      - predict in batches
      - upsert to predictions(id, model_signature, label, score?, predicted_at)
    Returns a summary list: [{"model_signature": ..., "predicted_rows": n}, ...]
    """
    from backend.ingestion import db as dal  # local import to avoid cycles
    dal.init_db()

    summary: list[dict[str, Any]] = []
    for clf in classifiers:
        model_sig = getattr(clf, "signature", clf.__class__.__name__)
        df_unlabeled = dal.fetch_unlabeled_for_model(model_sig)
        if df_unlabeled.empty:
            summary.append({"model_signature": model_sig, "predicted_rows": 0})
            continue

        total = 0
        for start in range(0, len(df_unlabeled), batch_size):
            block = df_unlabeled.iloc[start : start + batch_size]
            if merchant_col not in block.columns:
                raise KeyError(f"Column '{merchant_col}' not found in transactions.")
            labels = clf.predict(block[merchant_col])
            payload = [{"id": rid, "label": lab} for rid, lab in zip(block["id"], labels)]
            dal.upsert_predictions(model_sig, payload)
            total += len(block)

        summary.append({"model_signature": model_sig, "predicted_rows": total})
    return summary