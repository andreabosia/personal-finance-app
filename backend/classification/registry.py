from typing import Dict, List
from backend.classification.models.base import ClassifierStrategy

class StrategyRegistry:
    """
    Registry for classification strategies.
    Ensures unique names for each strategy.
    """
    def __init__(self):
        self._by_name: Dict[str, ClassifierStrategy] = {}
    def register(self, s: ClassifierStrategy):
        if s.name in self._by_name: raise ValueError(f"Strategy '{s.name}' already registered")
        self._by_name[s.name] = s
    def get(self, name: str) -> ClassifierStrategy:
        if name not in self._by_name: raise KeyError(f"Unknown strategy '{name}'")
        return self._by_name[name]
    @property
    def names(self) -> List[str]: return list(self._by_name.keys())