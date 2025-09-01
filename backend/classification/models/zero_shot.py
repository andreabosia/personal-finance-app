from dataclasses import dataclass, asdict, field
from typing import Any, List
from transformers import pipeline
from backend.classification.models.utils import ClassificationRequest, ClassificationResult, ClassScore
from backend.classification.models.base import ClassifierStrategy

@dataclass
class ZeroShotConfig:
    hf_model_id: str = "facebook/bart-large-mnli"
    candidate_labels: List[str] = field(default_factory=list)
    multi_label: bool = False
    hypothesis_template: str = "This text is about {}."

class ZeroShotClassifier(ClassifierStrategy):
    def __init__(self, cfg: ZeroShotConfig):
        if not cfg.candidate_labels: raise ValueError("candidate_labels must not be empty")
        self._cfg = cfg
        self._pipe = pipeline("zero-shot-classification", model=cfg.hf_model_id)

    @property
    def name(self) -> str: return "zero_shot"
    @property
    def config(self) -> Any: return asdict(self._cfg)

    def predict_one(self, req: ClassificationRequest) -> ClassificationResult:
        out = self._pipe(req.text, candidate_labels=self._cfg.candidate_labels,
                         hypothesis_template=self._cfg.hypothesis_template,
                         multi_label=self._cfg.multi_label)
        classes = [ClassScore(label=l, score=float(s)) for l, s in zip(out["labels"], out["scores"])]
        return ClassificationResult(strategy_name=self.name, classes=classes, raw=out)