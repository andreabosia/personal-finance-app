from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd, os

from backend.ingestion.extraction import TransactionExtractor

app = FastAPI()
CSV_PATH = "data/trusted/transactions.csv"

@app.get("/banks")
def list_banks():
    return {
        "banks": [
            {"key": key, "label": cls.DISPLAY_NAME}
            for key, cls in TransactionExtractor._BANK_MAP.items()
        ]
    }

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    bank: str = Form(...),
):
    content = await file.read()

    extractor = TransactionExtractor(bank=bank)

    df_new: pd.DataFrame = extractor.extract(content)
    if df_new.empty:
        return JSONResponse(content={"ok": False, "msg": f"No transactions found for bank={bank}"})

    if os.path.exists(CSV_PATH):
        df_existing = pd.read_csv(CSV_PATH, parse_dates=["data_operazione","data_valuta"])
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(CSV_PATH, index=False)

    return JSONResponse(
        content={
            "ok": True,
            "rows_added": len(df_new),
            "total_rows": len(df_all),
            "csv_path": CSV_PATH,
            "bank": bank,
        }
    )