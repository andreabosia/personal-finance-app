from dataclasses import dataclass
from typing import Optional, Dict, List
import re, io
import pdfplumber
import pandas as pd

@dataclass
class TransactionExtractorConfig:
    date_format: str = "%d.%m.%y"   # dd.mm.yy
    drop_empty_lines: bool = True
    keep_debug_cols: bool = False   # include left_gap/right_gap/raw_amount


class TransactionExtractor:
    """
    Extracts transactions from a PDF bank statement into a structured DataFrame.
    """

    _TXN_RE = re.compile(
        r"""
        ^\s*
        (?P<data_operazione>\d{2}\.\d{2}\.\d{2})   # first date
        \s+
        (?P<data_valuta>\d{2}\.\d{2}\.\d{2})       # second date
        (?P<left_gap>\s+)                          # spaces before amount
        (?P<amount>[0-9.'’ ]*,\d{2})               # amount
        (?P<right_gap>\s+)                         # spaces after amount
        (?P<descrizione>.+)$                       # description
        """,
        re.VERBOSE,
    )

    def __init__(self, config: Optional[TransactionExtractorConfig] = None):
        self.cfg = config or TransactionExtractorConfig()

    def extract(self, content: bytes) -> pd.DataFrame:
        """Main entrypoint: parse PDF bytes into structured transactions DataFrame."""
        rows = self._read_lines(content)
        df = pd.DataFrame(rows)
        return self._parse_lines_to_transactions(df)

    def _read_lines(self, content: bytes) -> List[Dict[str, str]]:
        lines_out: List[Dict[str, str]] = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                for ln in (page.extract_text_lines(layout=True) or []):
                    text = (ln.get("text") or "")
                    if self.cfg.drop_empty_lines and not text.strip():
                        continue
                    lines_out.append({"text": text})
        return lines_out

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

    def _parse_one_line(self, text: str) -> Optional[Dict[str, object]]:
        m = self._TXN_RE.match(text or "")
        if not m:
            return None
        gd = m.groupdict()
        left_gap, right_gap = len(gd["left_gap"]), len(gd["right_gap"])
        is_entrata = (left_gap > 1) or (left_gap > right_gap)
        amount_f = self._ita_to_float(gd["amount"])
        rec = {
            "data_operazione": gd["data_operazione"],
            "data_valuta": gd["data_valuta"],
            "uscite": None if is_entrata else amount_f,
            "entrate": amount_f if is_entrata else None,
            "descrizione": gd["descrizione"].strip(),
        }
        if self.cfg.keep_debug_cols:
            rec.update({
                "left_gap": left_gap,
                "right_gap": right_gap,
                "raw_amount": gd["amount"],
            })
        return rec

    def _parse_lines_to_transactions(self, df_lines: pd.DataFrame) -> pd.DataFrame:
        if df_lines.empty or "text" not in df_lines.columns:
            return pd.DataFrame(columns=["data_operazione","data_valuta","uscite","entrate","descrizione"])

        parsed = df_lines["text"].apply(self._parse_one_line)
        records = [r for r in parsed if r]
        out = pd.DataFrame(records)

        if out.empty:
            return out

        for c in ("data_operazione", "data_valuta"):
            out[c] = pd.to_datetime(out[c], format=self.cfg.date_format, errors="coerce")

        base_cols = ["data_operazione", "data_valuta", "uscite", "entrate", "descrizione"]
        debug_cols = ["left_gap", "right_gap", "raw_amount"]
        cols = base_cols + (debug_cols if self.cfg.keep_debug_cols else [])
        return out[[c for c in cols if c in out.columns]].reset_index(drop=True)