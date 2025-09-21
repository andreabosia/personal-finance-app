from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd, os

from backend.ingestion.extraction import TransactionExtractor
from backend.ingestion import db as dbdal

app = FastAPI()

# enpoint invoked to dinamically fill in drop down options for banks (i.e. differnet extarctor invoked depending on the bank)
@app.get("/banks")
def list_banks():
    return {
        "banks": [
            {"key": key, "label": cls.DISPLAY_NAME}
            for key, cls in TransactionExtractor._BANK_MAP.items()
        ]
    }

# endpoint to ingest a PDF file for a given bank
@app.post("/extract/{bank}")
async def ingest(bank: str, file: UploadFile):
    """Accept a PDF, parse it, save to SQLite."""
    if not file:
        raise HTTPException(status_code=400, detail="file is required (multipart/form-data)")
    # try:
    dbdal.init_db()
    content = await file.read()
    extractor = TransactionExtractor(bank)
    df = extractor.extract(content)
    extractor.save_to_db(df)
    return {"ok": True, "bank": bank, "ingested": int(len(df))}
    # except ValueError as e:
    #     raise HTTPException(status_code=400, detail=str(e))
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")