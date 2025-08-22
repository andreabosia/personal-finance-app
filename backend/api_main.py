# from fastapi import FastAPI 
# from pydantic import BaseModel
# from backend.model_inference import calculate
# class User_input(BaseModel) :
#     operation : str
#     x : float
#     y : float

# app = FastAPI()

# @app.post("/calculate")
# def operate(input:User_input):
#     result = calculate (input.operation, input.x, input.y)
#     return result

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import os

from backend.transaction_extractor import TransactionExtractor, TransactionExtractorConfig

app = FastAPI(title="Personal Finance Parser API")

extractor = TransactionExtractor(TransactionExtractorConfig())
CSV_PATH = "data/transactions.csv"
os.makedirs("data", exist_ok=True)

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    content = await file.read()
    df_new: pd.DataFrame = extractor.extract(content)

    if df_new.empty:
        return JSONResponse(content={"ok": False, "msg": "No transactions found"})

    # If CSV already exists, append new rows
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
        }
    )