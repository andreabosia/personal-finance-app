import streamlit as st
import requests
import pandas as pd
import numpy as np
import altair as alt
import os
from typing import Dict, List

EXTRACTOR_API_URL = "http://localhost:8000"      # extractor_api
CLASSIFIER_API_URL = "http://localhost:8001"     # classifier_api

# Models to request from the classifier API (must match your StrategyRegistry keys)
DEFAULT_MODELS = ["embedding", "zero_shot"]

# Where extractor saves the base CSV
CSV_PATH = "data/trusted/transactions.csv"
# Where we save the enriched CSV with categories merged in
ENRICHED_CSV_PATH = "data/trusted/transactions_enriched.csv"

st.set_page_config(page_title="Personal Finance App", layout="wide")

# -------------------------- Sidebar navigation --------------------------
page = st.sidebar.radio("Navigation", ["Upload PDF", "View Charts & Table"])


# ------------------------------- Utils ---------------------------------
@st.cache_data(ttl=3600)
def fetch_options_from_API(api_url: str, service: str):
    r = requests.get(f"{api_url}/{service}", timeout=5)
    r.raise_for_status()
    options = r.json()[f"{service}"]
    labels = [b["label"] for b in options]
    key_by_label = {b["label"]: b["key"] for b in options}
    return labels, key_by_label


def merge_and_save_categories(csv_path: str, cls_results: List[Dict]) -> str:
    """Merge classifier results (list of dicts) back to CSV and save enriched file."""
    if not os.path.exists(csv_path):
        return csv_path

    df_tx = pd.read_csv(csv_path, parse_dates=["data_operazione", "data_valuta"])
    res_df = pd.DataFrame(cls_results)

    # Make sure the expected columns exist
    if not {"id", "label"}.issubset(res_df.columns):
        return csv_path

    # Deduplicate per id: keep the highest score label if multiple strategies present
    res_df["score"] = pd.to_numeric(res_df.get("score", 0.0), errors="coerce").fillna(0.0)
    res_df = (
        res_df.sort_values(["id", "score"], ascending=[True, False])
              .drop_duplicates(subset=["id"], keep="first")
              [["id", "label"]]
              .rename(columns={"label": "category"})
    )

    # Ensure ID types align
    df_tx["id"] = df_tx["id"].astype(str)
    res_df["id"] = res_df["id"].astype(str)

    df_enriched = df_tx.merge(res_df, on="id", how="left")

    # Normalize category text
    if "category" in df_enriched.columns:
        df_enriched["category"] = df_enriched["category"].fillna("none").str.strip().str.lower()

    # Persist enriched CSV
    os.makedirs(os.path.dirname(ENRICHED_CSV_PATH), exist_ok=True)
    df_enriched.to_csv(ENRICHED_CSV_PATH, index=False)
    return ENRICHED_CSV_PATH


# ---------------- Page 1: Upload PDF, Extract CSV, Classify -------------
if page == "Upload PDF":
    st.title("Upload Bank Statement PDF")

    # Load extractor list (banks)
    try:
        extractor_labels, extractor_key_by_label = fetch_options_from_API(api_url=EXTRACTOR_API_URL, service="banks")
    except Exception as e:
        st.error(f"Could not load extractor options from backend: {e}")
        st.stop()

    bank_label = st.selectbox("Select bank", extractor_labels, index=None, placeholder="Choose a bank")

    # Let user pick which models to run (optional)
    with st.expander("Models to use for classification"):
        chosen_models = st.multiselect(
            "Select models",
            options=DEFAULT_MODELS,
            default=DEFAULT_MODELS
        )

    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    submit = st.button("Parse & Classify")

    if submit:
        if uploaded is None:
            st.warning("Please upload a PDF.")
            st.stop()
        if not bank_label:
            st.warning("Please select a bank.")
            st.stop()

        bank_key = extractor_key_by_label[bank_label]
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
        data = {"bank": bank_key}

        # ----- 1) Parse PDF via extractor API -----
        with st.spinner("Parsing PDF on backend..."):
            resp = requests.post(f"{EXTRACTOR_API_URL}/extract", files=files, data=data, timeout=120)

        if not resp.ok:
            st.error(f"Extractor failed (HTTP {resp.status_code}).")
            st.stop()

        payload = resp.json()
        if not payload.get("ok"):
            st.error(f"Extractor error: {payload}")
            st.stop()

        csv_path = payload.get("csv_path", CSV_PATH)
        st.success(
            f"✅ Added {payload.get('rows_added', 0)} rows. "
            f"Total rows: {payload.get('total_rows', 0)}.\n\n"
            f"Saved to: {csv_path}"
        )

        # ----- 2) Classify via Classifier API (CSVJob JSON body) -----
        job = {
            "csv_path": csv_path,
            "models": chosen_models or DEFAULT_MODELS,
            "id_col": "id",
            "merchant_col": "descrizione" if "descrizione" in pd.read_csv(csv_path, nrows=1).columns else "merchant",
            "amount_col": "ammontare",  # optional, ignored by your endpoint but OK to send
        }

        with st.spinner("Classifying transactions on backend..."):
            cls_resp = requests.post(f"{CLASSIFIER_API_URL}/classify/csv", json=job, timeout=180)

        if not cls_resp.ok:
            st.error(f"Classifier failed (HTTP {cls_resp.status_code}).")
            st.stop()

        cls_payload = cls_resp.json()
        st.info(
            f"🧠 Classified new rows: {cls_payload.get('classified_new_rows', 0)} "
            f"with strategies: {', '.join([s['name'] for s in cls_payload.get('strategies', [])])}"
        )

        # ----- 3) Merge categories back into CSV & persist enriched file -----
        join_job = {
            "csv_path": csv_path,
            "id_col": "id",     # from your earlier normalization step
            "out_path": None               # let backend pick "<base>_enriched.csv"
        }
        with st.spinner("Joining classifications into CSV..."):
            join_resp = requests.post(f"{CLASSIFIER_API_URL}/classify/join", json=join_job, timeout=120)

        if not join_resp.ok:
            st.error(f"Join failed (HTTP {join_resp.status_code}): {join_resp.text}")
            st.stop()

        join_payload = join_resp.json()
        enriched_path = join_payload.get("enriched_path")
        if enriched_path:
            st.success(
                f"📁 Enriched CSV ready: {enriched_path}  "
                f"(matched {join_payload.get('tx_with_classification', 0)} of {join_payload.get('tx_in_csv', 0)} rows)"
            )
        else:
            st.warning("Join succeeded but no enriched_path was returned.")


# ---------------- Page 2: Charts & Table -----------------
else:
    st.title("Expenses by Month + Full Table")

    read_path = ENRICHED_CSV_PATH if os.path.exists(ENRICHED_CSV_PATH) else CSV_PATH
    if not os.path.exists(read_path):
        st.warning("No data available yet. Upload and classify a PDF first from 'Upload PDF'.")
        st.stop()

    # Read and clean
    # Note: parse_dates to enable monthly bucketing
    df = pd.read_csv(read_path, parse_dates=["data_operazione", "data_valuta"])
    if "ammontare" in df.columns:
        df["ammontare"] = pd.to_numeric(df["ammontare"], errors="coerce")
    else:
        df["ammontare"] = 0.0

    # Ensure category column exists (fallback)
    if "category" not in df.columns:
        df["category"] = "none"

    # Month bucket
    df["month"] = df["data_operazione"].dt.to_period("M").dt.to_timestamp()

    # Entrate vs Uscite
    df["type"] = np.where(df["ammontare"] >= 0, "entrate", "uscite")

    # ---------------- Chart 1: Entrate vs Uscite by month ----------------
    left, right = st.columns((2, 1))
    with left:
        df_monthly_type = (
            df.groupby(["month", "type"], as_index=False)["ammontare"]
              .sum()
              .sort_values("month")
        )

        color_scale_eu = alt.Scale(domain=["entrate", "uscite"], range=["green", "red"])
        chart_eu = (
            alt.Chart(df_monthly_type)
            .mark_bar()
            .encode(
                x=alt.X("yearmonth(month):T", title="Month"),
                y=alt.Y("ammontare:Q", title="Amount"),
                color=alt.Color("type:N", scale=color_scale_eu, legend=alt.Legend(title="Tipo")),
                tooltip=[
                    alt.Tooltip("yearmonth(month):T", title="Month"),
                    alt.Tooltip("type:N",  title="Tipo"),
                    alt.Tooltip("ammontare:Q", title="Amount"),
                ],
            )
        )
        st.subheader("Entrate (↑) e Uscite (↓) Mensili")
        st.altair_chart(chart_eu, use_container_width=True)

    with right:
        st.subheader("Quick Stats")
        st.metric("Rows", len(df))
        st.metric("Total Amount", f"{df['ammontare'].sum():,.2f}")
        st.metric("# Categories", df["category"].nunique())

    # ---------------- Chart: Amount per Category (Pie) ----------------
    st.subheader("Totale per Categoria")

    df_cat_total = (
        df.groupby("category", as_index=False)["ammontare"]
        .sum()
        .sort_values("ammontare", ascending=False)
    )

    chart_cat_pie = (
        alt.Chart(df_cat_total)
        .mark_arc()
        .encode(
            theta=alt.Theta("ammontare:Q", title="Total Amount"),
            color=alt.Color("category:N", legend=alt.Legend(title="Category")),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("ammontare:Q", title="Total Amount", format=",.2f"),
            ],
        )
    )

    st.altair_chart(chart_cat_pie, use_container_width=True)

    # ---------------- Table & Download ----------------
    st.subheader("All Transactions")
    df_show = df.sort_values(["data_operazione"]).reset_index(drop=True)[["data_valuta","ammontare","descrizione","category","score"]]
    st.dataframe(df_show, use_container_width=True)

    st.download_button(
        "Download CSV",
        df_show.to_csv(index=False),
        file_name=os.path.basename(read_path),
        mime="text/csv",
    )