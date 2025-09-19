import sqlite3
import os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from typing import List, Dict, Any, Optional

#NOTE wal jurnal mode allows concurrent reads and writes

DB_PATH = Path(os.getenv("DB_PATH", "/data/trusted/results.db")).resolve()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_conn():
    return sqlite3.connect(str(DB_PATH))

def init_db():
    with get_conn() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id TEXT PRIMARY KEY,
            ts_ingested TEXT NOT NULL,
            bank TEXT,
            data_operazione TEXT,
            data_valuta TEXT,
            ammontare REAL,
            descrizione TEXT
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT NOT NULL,
            model_signature TEXT NOT NULL,
            label TEXT NOT NULL,
            score REAL,
            predicted_at TEXT NOT NULL,
            PRIMARY KEY (id, model_signature)
        );
        """)

def upsert_transactions_df(df: pd.DataFrame):
    if df.empty: 
        return
    init_db()
    df = df.copy()
    df["data_operazione"] = pd.to_datetime(df["data_operazione"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["data_valuta"] = pd.to_datetime(df["data_valuta"], errors="coerce").dt.strftime("%Y-%m-%d")
    now = datetime.now(timezone.utc).isoformat()
    df["ts_ingested"] = now

    rows = df.to_dict(orient="records")
    with get_conn() as conn:
        for r in rows:
            conn.execute("""
                INSERT INTO transactions(id, ts_ingested, bank, data_operazione, data_valuta, ammontare, descrizione)
                VALUES(:id, :ts_ingested, :bank, :data_operazione, :data_valuta, :ammontare, :descrizione)
                ON CONFLICT(id) DO NOTHING;
            """, r)
        conn.commit()

def fetch_unlabeled_for_model(model_signature: str, limit: Optional[int] = None) -> pd.DataFrame:
    init_db()
    sql = """
    SELECT t.*
    FROM transactions t
    LEFT JOIN predictions p
      ON p.id = t.id AND p.model_signature = ?
    WHERE p.id IS NULL
    """
    if limit:
        sql += " LIMIT ?"
        params = (model_signature, limit)
    else:
        params = (model_signature,)
    with get_conn() as conn:
        return pd.read_sql(sql, conn, params=params)
    
def upsert_predictions(model_signature: str, preds: List[Dict[str, Any]]):
    if not preds: 
        return
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        for p in preds:
            conn.execute("""
                INSERT INTO predictions(id, model_signature, label, score, predicted_at)
                VALUES(:id, :model_signature, :label, :score, :predicted_at)
                ON CONFLICT(id, model_signature) DO UPDATE SET
                    label=excluded.label,
                    score=excluded.score,
                    predicted_at=excluded.predicted_at;
            """, {
                "id": p["id"],
                "model_signature": model_signature,
                "label": p["label"],
                "score": p.get("score"),
                "predicted_at": now,
            })
        conn.commit()
    
def join_with_predictions(model_signature: str) -> pd.DataFrame:
    init_db()
    sql = """
    SELECT t.*, p.label, p.score, p.predicted_at
    FROM transactions t
    INNER JOIN predictions p
      ON p.id = t.id
     AND p.model_signature = ?
    """
    with get_conn() as conn:
        return pd.read_sql(sql, conn, params=(model_signature,))
    

def list_model_signatures() -> pd.DataFrame:
    init_db()
    sql = """
    SELECT
      model_signature,
      COUNT(*) AS n_predictions,
      MAX(predicted_at) AS last_predicted_at
    FROM predictions
    GROUP BY model_signature
    ORDER BY last_predicted_at DESC NULLS LAST
    """
    with get_conn() as conn:
        return pd.read_sql(sql, conn)
    