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

def orchestrate_db(
    model_config_path: str,
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
    dal.init_db()
    classifiers = load_model_from_yaml(model_config_path)
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
            # ensure column exists
            if merchant_col not in block.columns:
                raise KeyError(f"Column '{merchant_col}' not found in transactions.")

            labels = clf.predict(block[merchant_col])  # expects pd.Series -> pd.Series/list
            # OPTIONAL: scores = clf.predict_proba(block[merchant_col])

            payload = [
                {"id": rid, "label": lab}
                for rid, lab in zip(block["id"], labels)
            ]
            dal.upsert_predictions(model_sig, payload)
            total += len(block)

        summary.append({"model_signature": model_sig, "predicted_rows": total})

    return summary

if __name__ == "__main__":
    # defaults: adjust to your repo paths or pass via CLI/env
    DEFAULT_YAML = os.environ.get(
        "MODEL_CONFIG_YAML",
        "/Users/andreabosia/Projects/personal-finance-app/backend/classification/artifacts/model_config.yaml",
    )
    out = orchestrate_db(DEFAULT_YAML)
    print(out)