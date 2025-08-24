from __future__ import annotations
from typing import Optional, Dict, List
from abc import ABC, abstractmethod
import re, io
import pdfplumber
import pandas as pd

# -------------------------
#NOTE Base Class is TransactionExtractor from which sublclss inherit (and implement the abstarct method). 
# Factory design pattern to hide complexity of which extractor to adopt depending of the pdf (i.e. who calls the api, API_main, need not to know that there is a fineco and a ubs subclass, only neet to pass the right param.)
# -------------------------
class TransactionExtractor(ABC):
    """
    Factory + Template:
      TransactionExtractor(bank) -> returns a concrete subclass based on bank.
      extract() -> _read_lines() -> subclass _parse_one_line() -> finalize.
    """

    _BANK_MAP = {}  # filled after subclasses are defined (otherwise would have not defined error here)

    def __new__(cls, bank, *args, **kwargs):
        bank_key = bank.strip().lower()
        subcls = cls._BANK_MAP.get(bank_key)
        if not subcls:
            raise ValueError(f"Unsupported bank: {bank!r}. Supported: {sorted(cls._BANK_MAP)}")
        return super().__new__(subcls)

    def __init__(self, bank: str):
        self.bank = bank.strip().lower()

    # ----  Public ----
    def extract(self, content: bytes) -> pd.DataFrame:
        df_lines = self._read_lines(content)
        return self._parse_lines_to_transactions(df_lines)

    # ---- Private ----
    def _read_lines(self, content: bytes) -> pd.DataFrame:
        lines_out: List[Dict[str, str]] = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                for ln in (page.extract_text_lines(layout=True) or []):
                    text = (ln.get("text") or "")
                    if not text.strip():
                        continue
                    lines_out.append({"text": text})
        return pd.DataFrame(lines_out)

    @abstractmethod
    def _parse_one_line(self, text: str) -> Optional[Dict[str, object]]:
        ...

    @staticmethod
    def _ita_to_float(s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        x = (
            str(s)
            .replace("’", "")
            .replace("'", "")
            .replace(".", "")
            .replace("\u00a0", "")
            .replace(" ", "")
            .replace(",", ".")
        )
        try:
            return float(x)
        except Exception:
            return None

    def _parse_lines_to_transactions(self, df_lines: pd.DataFrame) -> pd.DataFrame:
        if df_lines.empty or "text" not in df_lines.columns:
            return pd.DataFrame(columns=["data_operazione","data_valuta","uscite","entrate","descrizione", "bank"])

        parsed = df_lines["text"].apply(self._parse_one_line)
        records = [r for r in parsed if r]
        out = pd.DataFrame(records)

        if out.empty:
            return out

        for c in ("data_operazione", "data_valuta"):
            if c in out.columns:
                out[c] = pd.to_datetime(out[c], format="%d.%m.%y", errors="coerce")

        base_cols = ["data_operazione", "data_valuta", "uscite", "entrate", "descrizione"]
        cols = [c for c in base_cols]
        out[cols].reset_index(drop=True)
        out[self.bank] = self.bank 
        return out


# -------------------------
# Fineco strategy
# -------------------------
class TransactionExtractorFineco(TransactionExtractor):
    DISPLAY_NAME = "Fineco"

    _TXN_RE = re.compile(
        r"""
        ^\s*
        (?P<data_operazione>\d{2}\.\d{2}\.\d{2})
        \s+
        (?P<data_valuta>\d{2}\.\d{2}\.\d{2})
        (?P<left_gap>\s+)
        (?P<amount>[0-9.'’ ]*,\d{2})
        (?P<right_gap>\s+)
        (?P<descrizione>.+)$   
        """,
        re.VERBOSE,
    )
    _DESCR_CLEAN_RE = re.compile(
        r" ?Carta N\.\s*\*+\s*\d+\s*Data Operazione\s*\d{2}/\d{2}/\d{2}", 
        re.IGNORECASE
    )    

    def _parse_one_line(self, text: str) -> Optional[Dict[str, object]]:
        m = self._TXN_RE.match(text or "")
        if not m:
            return None
        gd = m.groupdict()
        left_gap, right_gap = len(gd["left_gap"]), len(gd["right_gap"])
        is_entrata = (left_gap > 1) or (left_gap > right_gap)
        amount_f = self._ita_to_float(gd["amount"])

        # clean description
        descrizione = gd["descrizione"].strip()
        descrizione = self._DESCR_CLEAN_RE.sub("", descrizione).strip()
            
        rec = {
            "data_operazione": gd["data_operazione"],
            "data_valuta": gd["data_valuta"],
            "uscite": None if is_entrata else amount_f,
            "entrate": amount_f if is_entrata else None,
            "descrizione": descrizione,
        }
        return rec


# -------------------------
# UBS strategy (#TODO)
# -------------------------
class TransactionExtractorUBS(TransactionExtractor):
    DISPLAY_NAME = "UBS"

    _TXN_RE = re.compile(r"")

    def _parse_one_line(self, text: str) -> Optional[Dict[str, object]]:
        return


# ---- Register subclasses in the factory map ----
TransactionExtractor._BANK_MAP = {
    "fineco": TransactionExtractorFineco,
    "ubs": TransactionExtractorUBS,
}