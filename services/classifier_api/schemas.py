from typing import List, Optional
from pydantic import BaseModel

class CSVJob(BaseModel):
    csv_path: str
    models: List[str]
    id_col: str = "transaction_id"
    merchant_col: str = "merchant"
    amount_col: Optional[str] = "amount"

class ClassResult(BaseModel):
    transaction_id: str
    strategy: str
    signature: str
    label: Optional[str]
    score: Optional[float]
    ts: str
    raw_json: Optional[str]

class JoinJob(BaseModel):
    csv_path: str
    id_col: str = "transaction_id"
    # where to write the enriched file; if None, we will auto-generate in same folder
    out_path: Optional[str] = None