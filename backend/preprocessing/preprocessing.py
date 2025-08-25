# merchant → embedding → similarity features

# backend/preprocessing/preprocessing.py
from __future__ import annotations

import os
import time
import json
import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------
# Config & helpers
# -----------------------

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM_BY_MODEL = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
}

def now_ts() -> int:
    return int(time.time())

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def normalize_text(s: str) -> str:
    # light normalization; keep accents, but trim & lower
    return (s or "").strip().lower()

def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    A: [n, d] (assumed L2-normalized)
    B: [m, d] (assumed L2-normalized)
    returns [n, m] cosine similarity matrix
    """
    return A @ B.T


# -----------------------
# Data contracts
# -----------------------

@dataclass
class Paths:
    merchant_emb_parquet: str          # e.g., "artifacts/features/merchant_embeddings.parquet"
    category_emb_parquet: str          # e.g., "artifacts/features/category_embeddings.parquet"
    sims_parquet: str                  # e.g., "artifacts/features/tx_category_sims.parquet"


# -----------------------
# Main Preprocessor
# -----------------------

class MerchantCategoryPreprocessor:
    """
    - Embeds merchant strings from transactions and appends to Parquet (idempotent).
    - Embeds category descriptions and appends to Parquet (idempotent).
    - Computes top-K cosine similarities between merchant embeddings and category embeddings
      and appends to Parquet (idempotent).

    Assumptions:
      - You pass transactions with a stable `transaction_id` (string/uuid/int) and a `merchant_text` column.
      - You pass categories with a stable `category_id` and `category_text` column.
    """

    def __init__(
        self,
        paths: Paths,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,  # "cpu" or "cuda"; None lets sentence-transformers decide
        top_k: int = 5,
    ):
        self.paths = paths
        self.model_name = model_name
        self.top_k = int(top_k)

        # lazy-load model once
        self._model_obj: Optional[SentenceTransformer] = None
        self._device = device

        # Ensure dirs exist
        for p in (paths.merchant_emb_parquet, paths.category_emb_parquet, paths.sims_parquet):
            ensure_dir(p)


    # ------------- Public API -------------

    def embed_and_append_categories(
        self,
        categories_df: pd.DataFrame,
        category_id_col: str = "category_id",
        category_text_col: str = "category_text",
        language_col: Optional[str] = None,  # can be None; multilingual model handles it
    ) -> pd.DataFrame:
        """
        Creates/updates the category embeddings parquet.
        Skips rows already present for (category_id, model_name).
        """
        assert category_id_col in categories_df.columns
        assert category_text_col in categories_df.columns

        cat_df = categories_df.copy()
        cat_df[category_text_col] = cat_df[category_text_col].map(normalize_text)

        # Load existing ids for dedup
        existing = self._read_parquet_or_empty(self.paths.category_emb_parquet)
        existing_key = set()
        if not existing.empty:
            for cid, mname in existing[[category_id_col, "model_name"]].itertuples(index=False):
                existing_key.add((cid, mname))

        to_embed = []
        for row in cat_df.itertuples(index=False):
            cid = getattr(row, category_id_col)
            ctext = getattr(row, category_text_col)
            if (cid, self.model_name) in existing_key:
                continue
            to_embed.append((cid, ctext, getattr(row, language_col) if language_col else None))

        if not to_embed:
            return existing  # nothing new

        texts = [t[1] for t in to_embed]
        embs = self._encode(texts)  # [n, d], L2-normalized

        out_df = pd.DataFrame({
            category_id_col: [t[0] for t in to_embed],
            category_text_col: texts,
            "language": [t[2] for t in to_embed],
            "embedding": list(embs),  # store as list[float]
            "embedding_dim": embs.shape[1],
            "model_name": self.model_name,
            "created_at": now_ts(),
        })

        self._append_parquet(self.paths.category_emb_parquet, out_df)
        return self._read_parquet_or_empty(self.paths.category_emb_parquet)

    def embed_and_append_merchants(
        self,
        transactions_df: pd.DataFrame,
        transaction_id_col: str = "transaction_id",
        merchant_text_col: str = "descrizione",
    ) -> pd.DataFrame:
        """
        Creates/updates the merchant embeddings parquet. Skips already embedded transactions
        for (transaction_id, model_name). Returns the full merchant embedding table.
        """
        assert transaction_id_col in transactions_df.columns
        assert merchant_text_col in transactions_df.columns

        tx_df = transactions_df[[transaction_id_col, merchant_text_col]].copy()
        tx_df[merchant_text_col] = tx_df[merchant_text_col].map(normalize_text)

        existing = self._read_parquet_or_empty(self.paths.merchant_emb_parquet)
        existing_key = set()
        if not existing.empty:
            for tid, mname in existing[[transaction_id_col, "model_name"]].itertuples(index=False):
                existing_key.add((tid, mname))

        to_embed_rows = []
        for row in tx_df.itertuples(index=False):
            tid = getattr(row, transaction_id_col)
            mtext = getattr(row, merchant_text_col)
            if (tid, self.model_name) in existing_key:
                continue
            to_embed_rows.append((tid, mtext))

        if to_embed_rows:
            texts = [t[1] for t in to_embed_rows]
            embs = self._encode(texts)  # L2-normalized
            out_df = pd.DataFrame({
                transaction_id_col: [t[0] for t in to_embed_rows],
                merchant_text_col: texts,
                "embedding": list(embs),
                "embedding_dim": embs.shape[1],
                "model_name": self.model_name,
                "created_at": now_ts(),
                # Optional: deterministic content-hash of merchant_text for extra idempotency
                "merchant_text_hash": [sha256_text(t) for t in texts],
            })
            self._append_parquet(self.paths.merchant_emb_parquet, out_df)

        return self._read_parquet_or_empty(self.paths.merchant_emb_parquet)

    def compute_and_append_similarities(
        self,
        top_k: Optional[int] = None,
        transaction_id_col: str = "transaction_id",
        merchant_text_col: str = "descrizione",
        category_id_col: str = "category_id",
        category_text_col: str = "category_text",
    ) -> pd.DataFrame:
        """
        Computes top-K cosine similarities between *existing* merchant embeddings and category embeddings
        for the current model_name, and appends ONLY the missing (transaction_id, model_name) rows.

        Output Parquet schema (wide, but compact):
          - transaction_id
          - model_name
          - topk_category_ids: List[str]
          - topk_scores: List[float]
          - created_at
        """
        top_k = int(top_k or self.top_k)

        # Load tables filtered to current model_name
        merch = self._read_parquet_or_empty(self.paths.merchant_emb_parquet)
        cats = self._read_parquet_or_empty(self.paths.category_emb_parquet)

        merch = merch[merch["model_name"] == self.model_name].copy()
        cats = cats[cats["model_name"] == self.model_name].copy()

        if merch.empty or cats.empty:
            return self._read_parquet_or_empty(self.paths.sims_parquet)

        # Existing sims => skip already computed (transaction_id, model_name)
        sims_existing = self._read_parquet_or_empty(self.paths.sims_parquet)
        done_key = set()
        if not sims_existing.empty:
            for tid, mname in sims_existing[["transaction_id", "model_name"]].itertuples(index=False):
                done_key.add((tid, mname))

        # Build matrices
        # Deduplicate merchants by (transaction_id) and keep first
        merch_unique = merch.drop_duplicates(subset=[transaction_id_col])
        # Prepare arrays
        M = np.stack(merch_unique["embedding"].to_list(), axis=0).astype("float32")
        C = np.stack(cats["embedding"].to_list(), axis=0).astype("float32")

        # Matrix sims
        S = cosine_sim(M, C)  # [n_tx, n_cat]

        # For each merchant row, extract top-K
        topk_indices = np.argpartition(-S, kth=min(top_k, S.shape[1]-1), axis=1)[:, :top_k]
        # Re-sort those K by score desc for pretty outputs
        rows = []
        for i, tx_row in enumerate(merch_unique.itertuples(index=False)):
            tid = getattr(tx_row, transaction_id_col)
            if (tid, self.model_name) in done_key:
                continue
            idxs = topk_indices[i]
            # sort by score
            ordering = np.argsort(-S[i, idxs])
            idxs = idxs[ordering]
            scores = S[i, idxs].astype(float).tolist()
            cat_ids = cats.iloc[idxs][category_id_col].astype(str).tolist()
            rows.append({
                "transaction_id": tid,
                "model_name": self.model_name,
                "topk_category_ids": cat_ids,
                "topk_scores": scores,
                "created_at": now_ts(),
            })

        if rows:
            out_df = pd.DataFrame(rows)
            self._append_parquet(self.paths.sims_parquet, out_df)

        return self._read_parquet_or_empty(self.paths.sims_parquet)

    # ------------- Internal utils -------------

    def _get_model(self) -> SentenceTransformer:
        if self._model_obj is None:
            self._model_obj = SentenceTransformer(self.model_name, device=self._device)
        return self._model_obj  

    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = [normalize_text(t) for t in texts]
        emb = self._get_model().encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        emb = emb.astype("float32")
        return l2_normalize(emb)

    @staticmethod
    def _read_parquet_or_empty(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_parquet(path)

    @staticmethod
    def _append_parquet(path: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        if os.path.exists(path):
            # append mode by concat + overwrite (safe & simple)
            old = pd.read_parquet(path)
            new = pd.concat([old, df], ignore_index=True)
            new.to_parquet(path, index=False)
        else:
            df.to_parquet(path, index=False)