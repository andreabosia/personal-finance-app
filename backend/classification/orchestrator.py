from typing import Any, Dict, List
from backend.classification.models.utils import ClassificationRequest
from backend.classification.registry import StrategyRegistry

class Orchestrator:
    def __init__(self, registry: StrategyRegistry):
        self._reg = registry

    def classify_text(self, text: str, models: List[str]) -> List[Dict[str, Any]]:
        rows = []
        req = ClassificationRequest(text=text)
        for m in models:
            s = self._reg.get(m)
            res = s.predict_one(req)
            top = res.classes[0] if res.classes else None
            rows.append({
                "strategy": s.name,
                "signature": s.signature,
                "label": top.label if top else None,
                "score": float(top.score) if top else None,
                "raw_json": res.raw,
            })
        return rows