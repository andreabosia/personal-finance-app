from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple #, Iterable
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.classification.models.base import ClassifierStrategy
import pandas as pd
"""
This module implements an embedding-based classifier strategy using pre-trained sentence transformers.
It defines the EmbeddingAnchorClassifier class, which encodes input texts and compares them to predefined
class anchors using cosine similarity. The configuration for the classifier is encapsulated in the EmbeddingAnchorConfig
dataclass.
e.g.
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedder_id", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--class_anchors", default {"groceries": ["Esselunga","Unes"], "restaurants": ["ristorante","trattoria"]}, type=json.loads)
    p.add_argument("--normalize", default=True, type=bool)
    p.add_argument("--aggregate", default="max")
    p.add_argument("--top_k_debug", default=5,type=int)
    args = p.parse_args()
    cfg = EmbeddingAnchorConfig(
                    embedder_id=args.embedder_id,
                    class_anchors=args.class_anchors,
                    normalize=args.normalize,
                    aggregate=args.aggregate,
                    top_k_debug=args.top_k_debug)
    model = EmbeddingAnchorClassifier(cfg)
    res = model.predict_one(ClassificationRequest(text="I bought groceries at Esselunga"))
if name == "__main__":
    main()
"""

@dataclass
class EmbeddingAnchorConfig:
    """
    Configuration for the EmbeddingAnchorClassifier.
    Instead of using a complex dictionary-based config, I use a dataclass for simplicity and type safety.
    Attributes:
        embedder_id: The identifier of the pre-trained sentence transformer model to use.
        class_anchors: A dictionary mapping class labels to lists of anchor texts.
        normalize: Whether to normalize embeddings to unit length.
        aggregate: Method to aggregate multiple anchor similarities ('max' or 'mean').
        top_k_debug: Number of top anchors to include in debug output.
    """
    embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    class_anchors: Dict[str, List[str]] = None
    normalize: bool = True
    aggregate: str = "max"  # or "mean"
    top_k_debug: int = 5

class EmbeddingAnchorClassifier(ClassifierStrategy):
    """
    An embedding-based classifier that uses pre-defined anchor texts for each class.
    It encodes the input text and computes cosine similarities to the anchor embeddings.
    The class with the highest similarity score is assigned to the input text.
    The object is initialized with an EmbeddingAnchorConfig instance.
    """
    def __init__(self, cfg: EmbeddingAnchorConfig):
        if not cfg.class_anchors: raise ValueError("class_anchors required")
        self._cfg = cfg
        self._model = SentenceTransformer(cfg.embedder_id)
        pairs: List[Tuple[str,str]] = [(lbl, t) for lbl, texts in cfg.class_anchors.items() for t in texts]
        self._labels = [p[0] for p in pairs]
        texts = [p[1] for p in pairs]
        self._anchors = np.array(self._model.encode(texts, normalize_embeddings=cfg.normalize))

    @property
    def name(self) -> str: return "embedding"
    @property
    def config(self) -> Any: return asdict(self._cfg)

    def _cosine(self, a, b):
        """
        Computes cosine similarity between two sets of vectors.
        Note: If vectors are normalised then dot product is equivalent to cosine similarity.
        Args:
            a: A 2D numpy array of shape (m, d)
            b: A 2D numpy array of shape (n, d)
        Returns:
            A 2D numpy array of shape (m, n) with cosine similarities."""
        if not self._cfg.normalize:
            a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
            b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
        return a @ b.T
    
    def predict(self, x: pd.Series) -> pd.Series:
        """
        Vectorized prediction for a Series of texts.
        Args:
            x: pandas.Series of input texts.
        Returns:
            pandas.Series of predicted class labels.
        """
        texts = x.tolist()
        Q = self._model.encode(texts, normalize_embeddings=self._cfg.normalize)  # (B, d)
        S = self._cosine(Q, self._anchors)  # (B, n_anchors)

        lbl_array = np.array(self._labels)
        unique_labels = list(self._cfg.class_anchors.keys())
        label_to_indices = {lbl: np.where(lbl_array == lbl)[0] for lbl in unique_labels}

        agg_cols = []
        for lbl in unique_labels:
            idx = label_to_indices[lbl]
            slice_ = S[:, idx] if idx.size > 0 else np.full((S.shape[0], 1), -np.inf)
            if self._cfg.aggregate == "mean":
                v = slice_.mean(axis=1)
            else:
                v = slice_.max(axis=1)
            agg_cols.append(v)
        M = np.vstack(agg_cols).T  # (B, L)

        pred_indices = np.argmax(M, axis=1)
        y = pd.Series([unique_labels[i] for i in pred_indices], index=x.index)
        return y
    