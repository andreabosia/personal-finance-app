from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class ClassificationRequest:
    """ A request to classify a single text item."""
    text: str
    context: Optional[Dict[str, Any]] = None

@dataclass
class ClassScore:
    """ A class label with an associated score."""
    label: str
    score: float

@dataclass
class ClassificationResult:
    """ The result of classifying a single text item."""
    strategy_name: str
    classes: List[ClassScore]
    raw: Optional[Dict[str, Any]] = None