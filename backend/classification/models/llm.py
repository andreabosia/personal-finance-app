# from dataclasses import dataclass, asdict, field
# from typing import Any, List
# import json, re
# from transformers import pipeline
# from backend.classification.models.utils import ClassificationRequest, ClassificationResult, ClassScore
# from backend.classification.models.base import ClassifierStrategy

# @dataclass
# class LLMConfig:
#     hf_chat_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
#     candidate_labels: List[str] = field(default_factory=list)
#     system_prompt: str = ("Classify the merchant into exactly one of the labels. "
#                           "Return JSON: {\"label\":\"...\",\"rationale\":\"...\"}.")
#     temperature: float = 0.0
#     max_new_tokens: int = 128

# class LLMClassifier(ClassifierStrategy):
#     def __init__(self, cfg: LLMConfig):
  