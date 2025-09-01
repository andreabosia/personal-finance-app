from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple #, Iterable
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.classification.models.utils import ClassificationRequest, ClassificationResult, ClassScore
from backend.classification.models.base import ClassifierStrategy
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

    def predict_one(self, req: ClassificationRequest) -> ClassificationResult:
        q = self._model.encode([req.text], normalize_embeddings=self._cfg.normalize)  # (1,d)
        sims = self._cosine(q, self._anchors)[0]
        buckets: Dict[str, List[float]] = {}
        for lbl, s in zip(self._labels, sims):
            buckets.setdefault(lbl, []).append(float(s))
        scores = {k: (max(v) if self._cfg.aggregate=="max" else float(np.mean(v))) for k,v in buckets.items()}
        score_per_class = [ClassScore(k, v) for k, v in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        top_idx = np.argsort(sims)[::-1][: self._cfg.top_k_debug]
        raw = {
            "top_k_anchors": [
                {"label": self._labels[i], "similarity": float(sims[i])} 
                for i in top_idx
                ]
            }
        return ClassificationResult(strategy_name=self.name, classes=score_per_class, raw=raw)
    
    # def predict_batch(self, reqs: Iterable[ClassificationRequest]) -> List[ClassificationResult]:
    #     req_list = list(reqs)
    #     if not req_list:
    #         return []

    #     # 1) Encode all texts at once
    #     texts = [r.text for r in req_list]
    #     Q = self._model.encode(texts, normalize_embeddings=self._cfg.normalize)  # (B, d)

    #     # 2) Cosine similarity against all anchors → (B, n_anchors)
    #     S = self._cosine(Q, self._anchors)

    #     # 3) Build label→anchor indices ON THE FLY (no precompute)
    #     lbl_array = np.array(self._labels)  # one entry per anchor
    #     unique_labels = list(self._cfg.class_anchors.keys())  # preserves config order
    #     label_to_indices = {lbl: np.where(lbl_array == lbl)[0] for lbl in unique_labels}

    #     # 4) Aggregate per label to get (B, L) matrix of scores
    #     agg_cols = []
    #     for lbl in unique_labels:
    #         idx = label_to_indices[lbl]
    #         # (B, n_lbl_anchors); guard for empty (shouldn't happen if config is valid)
    #         slice_ = S[:, idx] if idx.size > 0 else np.full((S.shape[0], 1), -np.inf)
    #         if self._cfg.aggregate == "mean":
    #             v = slice_.mean(axis=1)
    #         else:  # default "max"
    #             v = slice_.max(axis=1)
    #         agg_cols.append(v)

    #     M = np.vstack(agg_cols).T  # (B, L)

    #     # 5) Build per-request results (sorted classes + top-k anchors debug)
    #     results: List[ClassificationResult] = []
    #     for b in range(M.shape[0]):
    #         row = M[b]
    #         order = np.argsort(row)[::-1]
    #         classes = [ClassScore(unique_labels[j], float(row[j])) for j in order]

    #         sims_row = S[b]  # (n_anchors,)
    #         top_idx = np.argsort(sims_row)[::-1][: self._cfg.top_k_debug]
    #         raw = {
    #             "top_k_anchors": [
    #                 {"label": self._labels[i], "similarity": float(sims_row[i])}
    #                 for i in top_idx
    #             ]
    #         }

    #         results.append(ClassificationResult(strategy_name=self.name, classes=classes, raw=raw))

    #     return results