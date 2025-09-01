import streamlit as st
import requests
import pandas as pd
import numpy as np
import altair as alt
import os

EXTRACTOR_API_URL = "http://localhost:8000"      # extractor_api
CLASSIFIER_API_URL = "http://localhost:8002"     # classifier_api
DEFAULT_CATEGORIES = ["groceries", "restaurants", "transport", "utilities", "shopping", "none"]
# PREPROC_URL = "http://localhost:8001"      # preprocessing_api
# TODO should be a default + user override
CSV_PATH = "data/trusted/transactions.csv"  # where extractor saves/reads CSV

st.set_page_config(page_title="Personal Finance App", layout="wide")
# -----------------------------------------------------------------------------------------------------
# -------------------------- Sidebar navigation -------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
page = st.sidebar.radio("Navigation", ["Upload PDF", "View Charts & Table"])


# -----------------------------------------------------------------------------------------------------
# --------------------------------------- Utils -------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_options_from_API(api_url: str, service: str):
    r = requests.get(f"{api_url}/{service}", timeout=5)
    r.raise_for_status()
    options = r.json()[f"{service}"]
    labels = [b["label"] for b in options]
    # needed to map back label to correct class
    key_by_label = {b["label"]: b["key"] for b in options}
    return labels, key_by_label

# -----------------------------------------------------------------------------------------------------
# ----------------- Page 1: Upload PDF, Extract CSV, Classify transactions ----------------------------
# -----------------------------------------------------------------------------------------------------
if page == "Upload PDF":
    st.title("Upload Bank Statement PDF")
    # Fetch available extractors (depending on bank)
    # if none is available, we cannot proceed -> stop execution
    try:
        extractor_labels, extractor_key_by_label = fetch_options_from_API(api_url=EXTRACTOR_API_URL, service="banks")
    except Exception as e:
        st.error(f"Could not load extractor options from backend: {e}"); st.stop()
    # Fetch available transactions classifiers
    #TODO if none is available, do we want to proceed anyway instead?
    # try:
    #     classifier_labels, classifier_key_by_label = fetch_options_from_API(api_url=CLASSIFIER_API_URL, service="classifiers")
    # except Exception as e:
    #     st.error(f"Could not load extractor options from backend: {e}"); st.stop()


    # select bank to use for parsing
    bank_label = st.selectbox("Select bank", extractor_labels, index=None, placeholder="Choose a bank")
    # select model to use for classification
    # classifier_label = st.selectbox("Select classifer", classifier_labels, index=None, placeholder="Choose a classifer")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    submit = st.button("Parse PDF")

    if submit:
        if uploaded is None:
            st.warning("Please upload a PDF."); st.stop()
        if not bank_label:
            st.warning("Please select a bank."); st.stop()
        # if not classifier_label:
        #     st.warning("Please select a classifier."); st.stop()            

        bank_key = extractor_key_by_label[bank_label]
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
        data = {"bank": bank_key}

        with st.spinner("Parsing PDF on backend..."):
            # ----- Call the extractor API -----
            #TODO should be a try execpt with error handling
            resp = requests.post(f"{EXTRACTOR_API_URL}/extract", files=files, data=data, timeout=60)

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
        with st.spinner("Classifing transactions on backend..."):
            # ----- Call the classifier API -----
            #TODO should be a try execpt with error handling
            resp = requests.post(f"{CLASSIFIER_API_URL}/classify/csv", files=files, data=data, timeout=60)

        if not resp.ok:
            st.error(f"Backend request failed (HTTP {resp.status_code}).")        


# -----------------------------------------------------------------------------------------------------                
# ----------------- Page 2: Charts & Table ------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
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