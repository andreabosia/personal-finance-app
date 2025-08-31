from __future__ import annotations
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

class ResultsStore:
    """
    Store for classification results using SQLite.
    Each result is uniquely identified by (transaction_id, strategy, signature).
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
            CREATE TABLE IF NOT EXISTS classification_enriched (
                transaction_id TEXT NOT NULL,
                strategy       TEXT NOT NULL,
                signature      TEXT NOT NULL,
                label          TEXT,
                score          REAL,
                ts             TEXT NOT NULL,
                raw_json       TEXT,
                PRIMARY KEY (transaction_id, strategy, signature)
            )""")
            con.commit()

    def existing_keys(self, transaction_ids: List[str], pairs: List[Tuple[str,str]]) -> set[Tuple[str,str,str]]:
        if not transaction_ids or not pairs: return set()
        strategies = list({p[0] for p in pairs})
        signatures = list({p[1] for p in pairs})
        q = f"""
        SELECT transaction_id, strategy, signature
        FROM classification_enriched
        WHERE transaction_id IN ({",".join("?"*len(transaction_ids))})
          AND strategy IN ({",".join("?"*len(strategies))})
          AND signature IN ({",".join("?"*len(signatures))})
        """
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(q, (*transaction_ids, *strategies, *signatures)).fetchall()
        return set(rows)

    def upsert_rows(self, rows: List[Dict[str, Any]]):
        if not rows: return
        with sqlite3.connect(self.db_path) as con:
            con.executemany("""
            INSERT OR REPLACE INTO classification_enriched
            (transaction_id, strategy, signature, label, score, ts, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [(r["transaction_id"], r["strategy"], r["signature"], r.get("label"),
                   r.get("score"), r.get("ts"), r.get("raw_json")) for r in rows])
            con.commit()

    def fetch_results(self, transaction_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as con:
            if transaction_ids:
                q = f"""SELECT transaction_id,strategy,signature,label,score,ts,raw_json
                        FROM classification_enriched
                        WHERE transaction_id IN ({",".join("?"*len(transaction_ids))})"""
                cur = con.execute(q, (*transaction_ids,))
            else:
                cur = con.execute("""SELECT transaction_id,strategy,signature,label,score,ts,raw_json
                                     FROM classification_enriched""")
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]