from __future__ import annotations
from typing import Optional, Dict, List
from abc import ABC, abstractmethod
from backend.ingestion.db import upsert_transactions_df
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
    
    def save_to_db(self, df: pd.DataFrame) -> None:
        """
        Upsert transactions into SQLite (idempotent on id).
        """
        if df.empty:
            return
        # ensure required columns exist
        needed = {"id","bank","data_operazione","data_valuta","ammontare","descrizione"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        upsert_transactions_df(df)
    
    # ---- Private ----
    def _add_id_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a unique id column to the transactions DataFrame.
        The id is a SHA-256 hash of key transaction fields and the bank name,
        ensuring uniqueness across multiple uploads and banks.
        """
        if df.empty:
            df["id"] = []
            return df

        def make_id(row):
            parts = [
                str(row.get("data_operazione") or ""),
                str(row.get("data_valuta") or ""),
                str(row.get("ammontare") or ""),
                str(row.get("descrizione") or ""),
                self.bank,
            ]
            base = "|".join(parts)
            return hashlib.sha256(base.encode("utf-8")).hexdigest()

        df = df.copy()
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
            return pd.DataFrame(columns=["data_operazione","data_valuta","ammontare","descrizione", "bank"])

        parsed = df_lines["text"].apply(self._parse_one_line)
        records = [r for r in parsed if r]
        out = pd.DataFrame(records)

        if out.empty:
            return out
        # fomate dates
        for c in ("data_operazione", "data_valuta"):
            if c in out.columns:
                out[c] = pd.to_datetime(out[c], format="%d.%m.%y", errors="coerce")
        out["bank"] = self.bank
        base_cols = ["data_operazione", "data_valuta", "ammontare", "descrizione", "bank"]
        cols = [c for c in base_cols]
        out[cols].reset_index(drop=True)
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
            "ammontare": amount_f if is_entrata else -amount_f,
            "descrizione": descrizione,
        }
        return rec


# -------------------------
# UBS strategy (#TODO)
# -------------------------
# -------------------------
# UBS strategy (words-grid only, robust "1 387.00" handling)
# -------------------------
class TransactionExtractorUBS(TransactionExtractor):
    DISPLAY_NAME = "UBS"

    # Row detectors
    _DATE_RE          = re.compile(r"^\d{2}\.\d{2}\.\d{2}$")
    _DECIMAL_TOKEN_RE = re.compile(r"^\d+\.\d{2}$")      # e.g., 387.00
    _DIG3_RE          = re.compile(r"^\d{3}$")           # 3-digit group
    _DIG1TO3_RE       = re.compile(r"^\d{1,3}$")         # leading 1–3 digits

    # Cleaning patterns
    _PAN_PAT      = re.compile(r"\d{4}X{8}\d{4}\s+\d{2}/\d{2}")   # 4397XXXXXXXX1665 07/28
    _ON_PAT       = re.compile(r"\bON\s+\d{2}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2}.*", re.IGNORECASE)
    _AMOUNT_META  = re.compile(r"^\s*AMOUNT\s+CHF\b.*", re.IGNORECASE)
    _DCP_PREFIX   = re.compile(r"\bDEBIT\s+CARD\s+PAYMENT\b", re.IGNORECASE)

    # Rows to exclude entirely
    _OPENING_BAL  = re.compile(r"\bopening\s+balance\b", re.IGNORECASE)
    _CLOSING_BAL  = re.compile(r"\bclosing\s+balance\b", re.IGNORECASE)
    _BAL_OVERVIEW = re.compile(r"\bbalance\s+overview\b", re.IGNORECASE)

    # Sign hints (fallback)
    _CREDIT_HINTS = (
        "salary payment", "credit interest", "incoming", "refund", "reversal",
        "lohn", "gehalt"  # DE salary words that appear on UBS slips
    )
    _DEBIT_HINTS  = (
        "debit card payment", "e-banking payment order", "payment", "sepa",
        "lsv", "direct debit", "standing order"
)

    def __init__(self, bank: str):
        super().__init__(bank)

    # ---------- public ----------
    def extract(self, content: bytes) -> pd.DataFrame:
        rows = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                words = page.extract_words(x_tolerance=2, y_tolerance=2) or []
                words.sort(key=lambda w: (round(w["top"], 1), w["x0"]))
                grouped: List[List[dict]] = []
                for w in words:
                    if not grouped or abs(grouped[-1][0]["top"] - w["top"]) > 3:
                        grouped.append([w])
                    else:
                        grouped[-1].append(w)

                last_txn = None
                for rw in grouped:
                    rw.sort(key=lambda w: w["x0"])
                    texts = [w["text"] for w in rw if w.get("text")]
                    if not texts:
                        continue

                    first = texts[0].strip()
                    if self._DATE_RE.match(first):
                        # ---- find amount groups on the row (e.g., "1 387.00") ----
                        amt_groups = self._amount_groups_on_row(rw, gap_px=25)
                        if not amt_groups:
                            last_txn = None
                            continue
                        # rightmost is balance, second-rightmost is txn amount
                        amt_groups.sort(key=lambda g: g["x0"])
                        balance_grp = amt_groups[-1]
                        amount_grp  = amt_groups[-2] if len(amt_groups) >= 2 else None

                        balance_val = self._to_float_en(balance_grp["text"])
                        amount_val  = self._to_float_en(amount_grp["text"]) if amount_grp else None

                        # value date = rightmost date on row (if any)
                        dates_on_row = [(i, w["text"], w["x0"]) for i, w in enumerate(rw)
                                        if self._DATE_RE.match((w["text"] or "").strip())]
                        value_date = first
                        if len(dates_on_row) >= 2:
                            dates_on_row.sort(key=lambda x: x[2])
                            value_date = dates_on_row[-1][1]

                        # description: text between first date and left edge of amount/balance
                        left_x  = rw[0]["x1"]
                        right_x = amount_grp["x0"] if amount_grp else balance_grp["x0"]
                        parts = []
                        for w in rw:
                            t = (w["text"] or "").strip()
                            if left_x - 0.1 <= w["x0"] <= right_x + 0.1:
                                # exclude dates and numeric chunks (to avoid pulling in amount fragments)
                                if not self._DATE_RE.match(t) and not self._is_numericish_chunk(t):
                                    parts.append(t)
                        descr = self._clean_descr(" ".join(parts).strip())

                        # skip opening/closing/footer
                        if self._is_opening_or_closing(descr):
                            last_txn = None
                            continue

                        last_txn = {
                            "data_operazione": first,
                            "data_valuta": value_date or first,
                            "ammontare": amount_val if amount_val is not None else 0.0,  # sign later
                            "descrizione": descr,
                            "_balance": balance_val,
                        }
                        rows.append(last_txn)

                    else:
                        # continuation line → append unless noise
                        if not last_txn:
                            continue
                        cont_raw = " ".join(texts).strip()
                        if not cont_raw or self._is_noise_continuation(cont_raw):
                            continue
                        extra = self._clean_descr(cont_raw)
                        if extra and not self._is_opening_or_closing(extra):
                            last_txn["descrizione"] = (last_txn["descrizione"] + "; " + extra).strip("; ").strip()

        if not rows:
            return pd.DataFrame(columns=["data_operazione","data_valuta","ammontare","descrizione","bank","id"])

        # infer signs via balance delta where possible; fallback to keyword hints
        prev_bal: Optional[float] = None
        for r in rows:
            amt = float(r.get("ammontare") or 0.0)
            bal = r.get("_balance")
            if prev_bal is not None and bal is not None and amt is not None:
                plus_err  = abs((prev_bal + abs(amt)) - bal)
                minus_err = abs((prev_bal - abs(amt)) - bal)
                signed = abs(amt) if plus_err < minus_err else -abs(amt)
            else:
                signed = self._sign_from_keywords(r.get("descrizione", ""), abs(amt))
            r["ammontare"] = signed
            prev_bal = bal if bal is not None else prev_bal
            r.pop("_balance", None)

        out = pd.DataFrame(rows)
        for c in ("data_operazione","data_valuta"):
            if c in out.columns:
                out[c] = pd.to_datetime(out[c], format="%d.%m.%y", errors="coerce")
        out["bank"] = self.bank
        out = out[["data_operazione","data_valuta","ammontare","descrizione","bank"]].reset_index(drop=True)
        out = self._add_id_column(out)
        return out

    # ---------- amount grouping (handles "1 387.00", "12 345.00", etc.) ----------
    def _amount_groups_on_row(self, rw: List[dict], gap_px: float = 25.0) -> List[Dict[str, object]]:
        """
        For each decimal token (e.g., '387.00'), pull in left-adjacent 1–3 digit groups within gap_px
        to form a full amount like '1 387.00' or '12 345.00'. Returns groups with x0/x1/text.
        """
        groups = []
        # indices of tokens that are decimal numbers
        dec_idxs = [i for i, w in enumerate(rw) if self._DECIMAL_TOKEN_RE.match((w["text"] or "").strip())]
        for i in dec_idxs:
            text = (rw[i]["text"] or "").strip()     # e.g., '387.00'
            x0   = rw[i]["x0"]
            x1   = rw[i]["x1"]

            # walk left and prepend groups separated by small gap
            j = i - 1
            first_chunk = True  # first left chunk may be 1–3 digits, subsequent left chunks must be exactly 3 digits
            while j >= 0:
                tj = (rw[j]["text"] or "").strip()
                if first_chunk:
                    ok = bool(self._DIG1TO3_RE.match(tj))
                else:
                    ok = bool(self._DIG3_RE.match(tj))
                if not ok:
                    break
                gap = x0 - rw[j]["x1"]
                if gap > gap_px:
                    break
                text = f"{tj} {text}"
                x0 = rw[j]["x0"]
                first_chunk = False
                j -= 1

            groups.append({"x0": x0, "x1": x1, "text": text})
        return groups

    # ---------- helpers ----------
    @staticmethod
    def _to_float_en(s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        x = s.replace("’","").replace("'","").replace("\u00a0","").replace(" ","")
        try:
            return float(x)
        except Exception:
            return None

    def _is_numericish_chunk(self, t: str) -> bool:
        """Tokens we don't want in description (dates or number pieces)."""
        t = (t or "").strip()
        return bool(self._DECIMAL_TOKEN_RE.match(t) or self._DIG1TO3_RE.match(t) or self._DIG3_RE.match(t))

    def _is_noise_continuation(self, text: str) -> bool:
        if self._AMOUNT_META.match(text): return True
        if self._ON_PAT.match(text): return True
        if self._PAN_PAT.search(text): return True
        return False

    def _clean_descr(self, s: str) -> str:
        if not s:
            return s
        s = self._AMOUNT_META.sub("", s)
        s = self._ON_PAT.sub("", s)
        m = self._PAN_PAT.search(s)
        if m:
            s = s[m.end():]
        s = self._DCP_PREFIX.sub("", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip(" ;,-")

    def _is_opening_or_closing(self, text: str) -> bool:
        return bool(
            self._OPENING_BAL.search(text)
            or self._CLOSING_BAL.search(text)
            or self._BAL_OVERVIEW.search(text)
        )

    def _sign_from_keywords(self, descr: str, amount_abs: float) -> float:
        t = (descr or "").lower()
        credit_hit = any(k in t for k in self._CREDIT_HINTS)
        debit_hit  = any(k in t for k in self._DEBIT_HINTS)

        # Prioritize credit phrases over generic "payment"
        if credit_hit and not debit_hit:
            return amount_abs
        if debit_hit and not credit_hit:
            return -amount_abs
        if credit_hit and debit_hit:
            # Resolve ambiguity: favor credit if salary/credit-like terms present
            if any(k in t for k in ("salary", "credit", "incoming", "refund", "reversal", "lohn", "gehalt")):
                return amount_abs
            return -amount_abs

        # No hints → default to debit (conservative)
        return -amount_abs

    def _parse_one_line(self, text: str):  # abstract satisfaction
        return None
# ---- Register subclasses in the factory map ----
TransactionExtractor._BANK_MAP = {
    "fineco": TransactionExtractorFineco,
    "ubs": TransactionExtractorUBS,
}