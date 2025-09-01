from __future__ import annotations
from dataclasses import asdict
from typing import Any, Iterable, List
import json, hashlib
from backend.classification.models.utils import ClassificationRequest, ClassificationResult

"""
This module defines the base interface for classifier strategies used in the personal finance app's classification system.
It includes the ClassifierStrategy abstract base class, which specifies the required properties and methods that all
classifier strategies must implement. The module also provides a utility function for generating unique signatures
based on the strategy's name and configuration.
"""

def sha1_short(obj: Any) -> str:
    """
    Generates a short SHA-1 hash for the given object.
    Args 
        obj: The object to hash (can be a dict, dataclass, etc.)
    Returns: 
        A short SHA-1 hash string
    """
    s = json.dumps(obj, sort_keys=True, default=lambda o: asdict(o) if hasattr(o, "__dict__") else str(o))
    return hashlib.sha1(s.encode()).hexdigest()[:10]    # short hash for brevity (no need for full sha1)

class ClassifierStrategy:
    """
    Abstract base class for classifier strategies.
    All classifier strategies must implement the properties and methods defined here.
    """
    @property
    def name(self) -> str: raise NotImplementedError
    @property
    def config(self) -> Any: raise NotImplementedError
    @property
    def signature(self) -> str: return sha1_short({"strategy": self.name, "config": self.config})
    def predict_one(self, req: ClassificationRequest) -> ClassificationResult: raise NotImplementedError
    def predict_batch(self, reqs: Iterable[ClassificationRequest]) -> List[ClassificationResult]:
        return [self.predict_one(r) for r in reqs]
    
