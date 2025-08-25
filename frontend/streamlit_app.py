import streamlit as st
import requests
import pandas as pd
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

                # ---- NEW: call preprocessing API to embed + compute similarities ----
                # 1) try to get fresh transactions directly from extractor response
                tx = payload.get("transactions")

                # 2) fallback: re-read from CSV and build minimal schema
                if not tx:
                    try:
                        df = pd.read_csv(payload.get("csv_path", CSV_PATH))
                        # adapt these column names if yours differ
                        needed_cols = {"transaction_id", "descrizione"}
                        if not needed_cols.issubset(df.columns):
                            # try common alternatives
                            mapping = {}
                            if "id" in df.columns: mapping["transaction_id"] = "id"
                            if "merchant" in df.columns: mapping["descrizione"] = "descrizione"
                            if mapping:
                                df = df.rename(columns={v: k for k, v in mapping.items() if v in df.columns})
                        tx = df[["transaction_id", "descrizione"]].dropna().to_dict(orient="records")
                    except Exception as e:
                        st.error(f"Could not prepare transactions for preprocessing: {e}")
                        tx = None

                if tx:
                    pre_body = {
                        "transactions": tx,
                        # optionally send categories here if you want to update them now:
                        # "categories": [{"category_id": "...", "category_text": "..."}],
                        "compute_sims": True,
                        "top_k": 5,
                    }
                    with st.spinner("Computing embeddings & similarities..."):
                        try:
                            pre_resp = requests.post(f"{PREPROC_URL}/preprocess", json=pre_body, timeout=120)
                            if pre_resp.ok:
                                pre_payload = pre_resp.json()
                                st.success(f"Embeddings updated (model: {pre_payload.get('model_name')}).")
                            else:
                                st.warning(f"Preprocessing API returned HTTP {pre_resp.status_code}: {pre_resp.text}")
                        except Exception as e:
                            st.error(f"Preprocessing API call failed: {e}")
                else:
                    st.info("No transactions found to preprocess.")
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
    for col in ("uscite", "entrate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Chart: total expenses by month ----
    left, right = st.columns((2, 1))
    with left:
        df["month"] = df["data_operazione"].dt.to_period("M").dt.to_timestamp()
        df_monthly = (
            df.groupby("month", as_index=False)["uscite"]
              .sum()
              .sort_values("month")
        )

        chart = (
            alt.Chart(df_monthly)
            .mark_bar()
            .encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("uscite:Q", title="Total Expenses"),
                tooltip=[alt.Tooltip("month:T", title="Month"),
                         alt.Tooltip("uscite:Q", title="Expenses")]
            )
        )
        st.subheader("Total Expenses by Month")
        st.altair_chart(chart, use_container_width=True)

    # ---- Table: full CSV ----
    with right:
        st.subheader("Quick Stats")
        st.metric("Rows", len(df))
        st.metric("Total Expenses", f"{df['uscite'].sum():,.2f}")
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