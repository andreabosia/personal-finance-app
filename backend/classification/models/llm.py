from dataclasses import dataclass, asdict, field
from typing import Any, List
import json, re
from transformers import pipeline
from backend.classification.models.utils import ClassificationRequest, ClassificationResult, ClassScore
from backend.classification.models.base import ClassifierStrategy

@dataclass
class LLMConfig:
    hf_chat_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    candidate_labels: List[str] = field(default_factory=list)
    system_prompt: str = ("Classify the merchant into exactly one of the labels. "
                          "Return JSON: {\"label\":\"...\",\"rationale\":\"...\"}.")
    temperature: float = 0.0
    max_new_tokens: int = 128

class LLMClassifier(ClassifierStrategy):
    def __init__(self, cfg: LLMConfig):
        self._cfg = cfg
        self._pipe = pipeline("text-generation", model=cfg.hf_chat_model_id)

    @property
    def name(self) -> str: return "llm"
    @property
    def config(self) -> Any: return asdict(self._cfg)

    def _prompt(self, text: str) -> str:
        labels = ", ".join(self._cfg.candidate_labels) if self._cfg.candidate_labels else "N/A"
        return f"{self._cfg.system_prompt}\nLabels: [{labels}]\nMerchant: {text}\nAnswer strictly as JSON."

    def predict_one(self, req: ClassificationRequest) -> ClassificationResult:
        gen = self._pipe(self._prompt(req.text), max_new_tokens=self._cfg.max_new_tokens,
                         temperature=self._cfg.temperature, do_sample=False)
        txt = gen[0]["generated_text"]
        lbl, raw = "UNKNOWN", {"raw": txt}
        try:
            m = re.search(r"\{.*\}", txt, flags=re.S)
            if m:
                js = json.loads(m.group(0))
                lbl, raw = js.get("label", lbl), js
        except Exception:
            pass
        return ClassificationResult(strategy_name=self.name, classes=[ClassScore(lbl, 1.0)], raw=raw)