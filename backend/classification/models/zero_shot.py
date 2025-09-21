# from dataclasses import dataclass, asdict, field
# from typing import Any, List
# from transformers import pipeline
# from backend.classification.models.utils import ClassificationRequest, ClassificationResult, ClassScore
# from backend.classification.models.base import ClassifierStrategy

# @dataclass
# class ZeroShotConfig:
#     hf_model_id: str = "facebook/bart-large-mnli"
#     candidate_labels: List[str] = field(default_factory=list)
#     multi_label: bool = False
#     hypothesis_template: str = "Classify the merchnt into one of the possible categories {}."

# class ZeroShotClassifier(ClassifierStrategy):
#     def __init__(self, cfg: ZeroShotConfig):
