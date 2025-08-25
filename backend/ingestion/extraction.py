from __future__ import annotations
from typing import Optional, Dict, List
from abc import ABC, abstractmethod
import re, io
import pdfplumber
import pandas as pd
import hashlib


class TransactionExtractor(ABC):
    """
    Abstract base class for extracting transaction data from bank statement PDFs.
    Factory design pattern to hide complexity of which extractor to adopt depending of the pdf (i.e. who calls the api, need not to know that there is a fineco 
    and a ubs subclass, only neet to pass the right param.)

    Implements a Factory + Template design pattern:
      - Factory: TransactionExtractor(bank) returns a concrete subclass based on the bank name.
      - Template: extract() provides a standard workflow for reading and parsing PDF content,
        delegating line parsing to the subclass via the abstract _parse_one_line() method.

    Subclasses must implement _parse_one_line() to handle bank-specific parsing logic.
    The factory hides the complexity of which extractor to use, allowing clients to simply
    specify the bank name.
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
        df_txns = self._parse_lines_to_transactions(df_lines)
        df_txns = self._add_id_column(df_txns)
        return df_txns
    
    # ---- Private ----
    def _add_id_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a unique 'id' column to the DataFrame based on transaction fields.
        """
        if df.empty:
            df["id"] = []
            return df

        def make_id(row):
            # Use relevant fields to generate a unique hash
            base = f"{row.get('data_operazione','')}_{row.get('data_valuta','')}_{row.get('uscite','')}_{row.get('entrate','')}_{row.get('descrizione','')}_{self.bank}"
            return hashlib.sha256(base.encode("utf-8")).hexdigest()

        df["id"] = df.apply(make_id, axis=1)
        return df

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