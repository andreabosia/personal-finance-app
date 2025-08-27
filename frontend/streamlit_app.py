import streamlit as st
import requests
import pandas as pd
import numpy as np
import altair as alt
import os

BACKEND_URL = "http://localhost:8000"      # extractor_api
PREPROC_URL = "http://localhost:8001"      # preprocessing_api
CSV_PATH = "data/trusted/transactions.csv"

st.set_page_config(page_title="Personal Finance App", layout="wide")

# --- Sidebar navigation ---
page = st.sidebar.radio("Navigation", ["Upload PDF", "View Charts & Table"])

def fetch_banks():
    r = requests.get(f"{BACKEND_URL}/banks", timeout=5)
    r.raise_for_status()
    return r.json()["banks"]

@st.cache_data(ttl=3600)
def get_bank_choices():
    banks = fetch_banks()
    labels = [b["label"] for b in banks]
    key_by_label = {b["label"]: b["key"] for b in banks}
    return labels, key_by_label

# --- Page 1: Upload PDF ---
if page == "Upload PDF":
    st.title("Upload Bank Statement PDF")

    try:
        labels, key_by_label = get_bank_choices()
    except Exception as e:
        st.error(f"Could not load banks from backend: {e}")
        labels, key_by_label = ["Fineco", "UBS"], {"Fineco": "fineco", "UBS": "ubs"}  # fallback

    bank_label = st.selectbox("Select bank", labels, index=None, placeholder="Choose a bank")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    submit = st.button("Parse PDF")

    if submit:
        if uploaded is None:
            st.warning("Please upload a PDF."); st.stop()
        if not bank_label:
            st.warning("Please select a bank."); st.stop()

        bank_key = key_by_label[bank_label]
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
        data = {"bank": bank_key}

        with st.spinner("Parsing PDF on backend..."):
            resp = requests.post(f"{BACKEND_URL}/extract", files=files, data=data, timeout=60)

        if not resp.ok:
            st.error(f"Backend request failed (HTTP {resp.status_code}).")
        else:
            payload = resp.json()
            if payload.get("ok"):
                st.success(
                    f"✅ Added {payload.get('rows_added', 0)} rows. "
                    f"Total rows: {payload.get('total_rows', 0)}.\n\n"
                    f"Saved to: {payload.get('csv_path', CSV_PATH)}"
                )

                # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



            # ---- NEW: call LLM classifier API ----
            # expects:
            #   - CLASSIFIER_URL = "http://localhost:8000"  (or wherever your API runs)
            #   - categories: List[str] (e.g., from a widget)
            #   - optional llm settings (backend/model_name/temperature/max_new_tokens)

            CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://localhost:8002")
            DEFAULT_CATEGORIES = ["groceries", "restaurants", "transport", "utilities", "shopping", "none"]

            # you can wire this to Streamlit inputs; here we just fall back to defaults
            categories = DEFAULT_CATEGORIES#payload.get("categories") or DEFAULT_CATEGORIES
            backend = payload.get("backend", "ollama")        # or "hf"
            model_name = payload.get("model_name", "mistral") # e.g., "llama3.1:8b" for Ollama
            temperature = float(payload.get("temperature", 0.0))
            max_new_tokens = int(payload.get("max_new_tokens", 64))

            # 1) source transactions from extractor payload if present
            tx = payload.get("transactions")

            # 2) fallback: load from CSV
            if not tx:
                try:
                    df = pd.read_csv(payload.get("csv_path", CSV_PATH))
                    # Make sure we have these two columns; keep txn_id to reattach later
                    # Try to map common alternatives
                    if "transaction_id" not in df.columns and "id" in df.columns:
                        df = df.rename(columns={"id": "transaction_id"})
                    if "descrizione" not in df.columns and "merchant" in df.columns:
                        df = df.rename(columns={"merchant": "descrizione"})

                    need = {"descrizione"}
                    if not need.issubset(df.columns):
                        raise ValueError(f"CSV missing required columns {need}. Found: {list(df.columns)}")

                    # Build a thin list of records for the API (only descrizione is strictly required)
                    tx_df = df[["descrizione"]].copy()
                    # keep a local index to align results if API doesn’t echo ids
                    tx = tx_df.to_dict(orient="records")
                except Exception as e:
                    st.error(f"Could not prepare transactions for LLM classification: {e}")
                    tx = None

            if tx:
                req_body = {
                    "categories": categories,
                    "records": tx,                  # each item: {"descrizione": "..."}
                    "backend": backend,
                    "model_name": model_name,
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                }

                with st.spinner("Classifying transactions with LLM..."):
                    try:
                        resp = requests.post(f"{CLASSIFIER_URL}/classify_records", json=req_body, timeout=180)
                        if not resp.ok:
                            st.warning(f"Classifier API returned HTTP {resp.status_code}: {resp.text}")
                        else:
                            data = resp.json()
                            results = data.get("results", [])

                            if not results:
                                st.info("Classifier returned no results.")
                            else:
                                # Merge predictions back to your df.
                                # If you built tx from df[["descrizione"]], order matches; align by position.
                                preds = [r.get("predicted_category", "unknown") for r in results]

                                # Ensure we have df in scope (if extractor gave transactions, rebuild df from them)
                                if "df" not in locals():
                                    # If extractor gave tx with more fields, you can turn it into df.
                                    df = pd.DataFrame(payload["transactions"])

                                df = df.copy()
                                df["predicted_category"] = preds[:len(df)]  # safeguard against length mismatch

                                st.success(f"Classified {len(preds)} transactions with model '{model_name}'.")
                                st.dataframe(df.head(20))

                                # Optional: save to disk
                                out_path = payload.get("classified_csv_path", "artifacts/transactions_classified.csv")
                                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                                df.to_csv(out_path, index=False)
                                st.caption(f"Saved classified CSV → {out_path}")
                    except Exception as e:
                        st.error(f"Classifier API call failed: {e}")







                # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                # # ---- NEW: call preprocessing API to embed + compute similarities ----
                # # 1) try to get fresh transactions directly from extractor response
                # tx = payload.get("transactions")

                # # 2) fallback: re-read from CSV and build minimal schema
                # if not tx:
                #     try:
                #         df = pd.read_csv(payload.get("csv_path", CSV_PATH))
                #         # adapt these column names if yours differ
                #         needed_cols = {"transaction_id", "descrizione"}
                #         if not needed_cols.issubset(df.columns):
                #             # try common alternatives
                #             mapping = {}
                #             if "id" in df.columns: mapping["transaction_id"] = "id"
                #             if "merchant" in df.columns: mapping["descrizione"] = "descrizione"
                #             if mapping:
                #                 df = df.rename(columns={v: k for k, v in mapping.items() if v in df.columns})
                #         tx = df[["transaction_id", "descrizione"]].dropna().to_dict(orient="records")
                #     except Exception as e:
                #         st.error(f"Could not prepare transactions for preprocessing: {e}")
                #         tx = None

                # if tx:
                #     pre_body = {
                #         "transactions": tx,
                #         # optionally send categories here if you want to update them now:
                #         # "categories": [{"category_id": "...", "category_text": "..."}],
                #         "compute_sims": True,
                #         "top_k": 5,
                #     }
                #     with st.spinner("Computing embeddings & similarities..."):
                #         try:
                #             pre_resp = requests.post(f"{PREPROC_URL}/preprocess", json=pre_body, timeout=120)
                #             if pre_resp.ok:
                #                 pre_payload = pre_resp.json()
                #                 st.success(f"Embeddings updated (model: {pre_payload.get('model_name')}).")
                #             else:
                #                 st.warning(f"Preprocessing API returned HTTP {pre_resp.status_code}: {pre_resp.text}")
                #         except Exception as e:
                #             st.error(f"Preprocessing API call failed: {e}")
                # else:
                #     st.info("No transactions found to preprocess.")
                # ---------------------------------------------------------------------

            else:
                st.error(payload.get("msg", payload.get("error", "Unknown error")))
                
# --- Page 2: Charts & Table ---
else:
    st.title("Expenses by Month + Full Table")

    if not os.path.exists(CSV_PATH):
        st.warning("No data available yet. Upload a PDF first from 'Upload PDF'.")
        st.stop()

    # Read and clean
    df = pd.read_csv(CSV_PATH, parse_dates=["data_operazione", "data_valuta"])

    # Ensure numeric columns (if the CSV was created elsewhere)
    for col in ("ammontare"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

# ---- Chart: Entrate (green) vs Uscite (red) by month ----
left, right = st.columns((2, 1))
with left:
    # Month bucket
    df["month"] = df["data_operazione"].dt.to_period("M").dt.to_timestamp()

    # Label type from sign
    df["type"] = np.where(df["ammontare"] >= 0, "entrate", "uscite")

    # Aggregate by month and type
    df_monthly = (
        df.groupby(["month", "type"], as_index=False)["ammontare"]
          .sum()
          .sort_values("month")
    )

    # Color map
    color_scale = alt.Scale(domain=["entrate", "uscite"], range=["green", "red"])

    chart = (
        alt.Chart(df_monthly)
        .mark_bar()
        .encode(
            # force x axis granularity to months
            x=alt.X("yearmonth(month):T", title="Month"),
            y=alt.Y("ammontare:Q", title="Amount"),
            color=alt.Color("type:N", scale=color_scale, legend=alt.Legend(title="Tipo")),
            tooltip=[
                alt.Tooltip("yearmonth(month):T", title="Month"),
                alt.Tooltip("type:N",  title="Tipo"),
                alt.Tooltip("ammontare:Q", title="Amount"),
            ],
        )
    )

    st.subheader("Entrate (↑) e Uscite (↓) Mensili")
    st.altair_chart(chart, use_container_width=True)

    # ---- Table: full CSV ----
    with right:
        st.subheader("Quick Stats")
        st.metric("Rows", len(df))
        st.metric("Total Expenses", f"{df['ammontare'].sum():,.2f}")
        if "entrate" in df.columns:
            st.metric("Total Incomes", f"{df['entrate'].sum():,.2f}")

    st.subheader("All Transactions (from CSV)")
    # Nice sorting defaults
    df_show = df.sort_values(["data_operazione", "descrizione"]).reset_index(drop=True)
    st.dataframe(df_show, use_container_width=True)

    # Download button
    st.download_button(
        "Download CSV",
        df_show.to_csv(index=False),
        file_name="transactions.csv",
        mime="text/csv",
    )