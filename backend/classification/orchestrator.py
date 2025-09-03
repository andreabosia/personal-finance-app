from typing import Iterable, Dict, Any
import pandas as pd
from backend.classification import db
# strategy must expose: predict(series) -> Series (aligned)

def run_classification_for_model(model_signature: str, clf, *, text_col="descrizione", batch_size: int = 512) -> int:
    df_unlabeled = db.fetch_unlabeled_for_model(model_signature)
    if df_unlabeled.empty:
        return 0

    total = 0
    for start in range(0, len(df_unlabeled), batch_size):
        block = df_unlabeled.iloc[start:start+batch_size]
        labels = clf.predict(block[text_col])                # pd.Series
        payload = [{"id": rid, "label": lab}
                   for rid, lab in zip(block["id"], labels)]
        db.upsert_predictions(model_signature, payload)
        total += len(block)
    return total