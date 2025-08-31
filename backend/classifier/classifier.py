# classifier.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Any, Literal
import time
import json
import re
import random

import pandas as pd

# Optional imports: only required if you use the corresponding backend
try:
    import requests  # for Ollama
except Exception:
    requests = None

try:
    from transformers import pipeline  # for Hugging Face
except Exception:
    pipeline = None


Backend = Literal["ollama", "hf"]


@dataclass
class LLMConfig:
    backend: Backend = "ollama"                # "ollama" or "hf"
    model_name: str = "llama3.1:8b"            # ollama tag OR HF model id / local path
    temperature: float = 0.0
    top_p: float = 0.9
    max_new_tokens: int = 64
    # HuggingFace-specific:
    hf_device: Optional[str] = None            # e.g., "cpu" or "cuda:0"
    hf_dtype: Optional[str] = None             # e.g., "float16"
    # Ollama-specific:
    ollama_url: str = "http://localhost:11434"


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._pipe = None

    def _ensure_hf_pipeline(self):
        if pipeline is None:
            raise RuntimeError("transformers not installed. `pip install transformers accelerate torch`")
        if self._pipe is None:
            kwargs = {
                "task": "text-generation",
                "model": self.cfg.model_name,
            }
            if self.cfg.hf_device:
                kwargs["device"] = 0 if "cuda" in self.cfg.hf_device else -1
            self._pipe = pipeline(**kwargs)

    def generate(self, prompt: str) -> str:
        """
        Returns raw text from the model.
        """
        if self.cfg.backend == "ollama":
            if requests is None:
                raise RuntimeError("requests not installed. `pip install requests`")
            payload = {
                "model": self.cfg.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": self.cfg.temperature,
                    "top_p": self.cfg.top_p,
                    "num_predict": self.cfg.max_new_tokens,
                },
                "stream": False,
            }
            r = requests.post(f"{self.cfg.ollama_url}/api/generate", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "")

        elif self.cfg.backend == "hf":
            self._ensure_hf_pipeline()
            outs = self._pipe(
                prompt,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=(self.cfg.temperature > 0.0),
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                return_full_text=False,
            )
            # HF returns a list of generations; take the first
            if isinstance(outs, list) and outs:
                text = outs[0].get("generated_text", "")
            else:
                text = str(outs)
            return text

        else:
            raise ValueError(f"Unknown backend: {self.cfg.backend}")


# -----------------------------
# Classification prompt & logic
# -----------------------------

SYSTEM_RULES = """You are a strict classifier. You must pick exactly ONE category from the allowed list.
Return ONLY the category id, nothing else. If nothing fits, return 'unknown'."""

PROMPT_TEMPLATE = """{system_rules}

Allowed categories (ids):
{category_block}

Transaction description:
{description}

Return exactly ONE line containing only the category id from the allowed list above (or 'unknown').
"""

def build_prompt(description: str, categories: List[str]) -> str:
    # Present categories one per line to reduce parsing mistakes
    cat_block = "\n".join(f"- {c}" for c in categories)
    return PROMPT_TEMPLATE.format(
        system_rules=SYSTEM_RULES,
        category_block=cat_block,
        description=description.strip() if description else "",
    )


def clean_to_allowed(answer: str, allowed: List[str]) -> str:
    """
    Normalize the model’s answer to one of the allowed categories,
    otherwise 'unknown'.
    """
    if not answer:
        return "unknown"

    text = answer.strip().strip('"').strip("'")
    text = re.split(r"[\r\n]", text)[0].strip()  # first line only

    # Exact match first
    if text in allowed:
        return text

    # Case-insensitive exact
    lowered = {c.lower(): c for c in allowed}
    if text.lower() in lowered:
        return lowered[text.lower()]

    # Heuristic: sometimes models return like "Category: groceries"
    m = re.search(r"([A-Za-z0-9_\- ]+)$", text)
    if m:
        candidate = m.group(1).strip()
        if candidate in allowed:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return "unknown"


def classify_one(description: str, categories: List[str], llm: LLMClient, retries: int = 2, backoff: float = 0.6) -> str:
    allowed = categories or []
    if not allowed:
        return "unknown"

    prompt = build_prompt(description, allowed)

    for attempt in range(retries + 1):
        try:
            raw = llm.generate(prompt)
            label = clean_to_allowed(raw, allowed)
            if label != "unknown" or attempt == retries:
                return label
        except Exception:
            if attempt == retries:
                return "unknown"
        # backoff & small jitter
        time.sleep(backoff * (1.0 + random.random() * 0.25))

    return "unknown"


def classify_dataframe(
    df: pd.DataFrame,
    categories: List[str],
    description_col: str = "descrizione",
    out_col: str = "predicted_category",
    llm_cfg: Optional[LLMConfig] = None,
) -> pd.DataFrame:
    """
    Classify each row by feeding `descrizione` to the LLM, adding `predicted_category`.
    """
    if df.empty:
        df[out_col] = pd.Series(dtype="string")
        return df

    if description_col not in df.columns:
        raise ValueError(f"Column '{description_col}' not found in DataFrame. Available: {list(df.columns)}")

    cfg = LLMConfig(backend="ollama", model_name="llama3.1:8b")
    client = LLMClient(cfg)

    preds: List[str] = []
    for desc in df[description_col].astype(str).tolist():
        label = classify_one(desc, categories, client)
        preds.append(label)

    out = df.copy()
    out[out_col] = preds
    return out