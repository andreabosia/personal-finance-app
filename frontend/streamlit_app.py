# frontend/streamlit_app.py
import time
import streamlit as st
import requests
import pandas as pd
import numpy as np
import altair as alt
import os
from typing import Dict, List, Tuple

EXTRACTOR_API_URL = os.getenv("EXTRACTOR_API_URL", "http://localhost:8000")
CLASSIFIER_API_URL = os.getenv("CLASSIFIER_API_URL", "http://localhost:8001")

# Your default YAML path (the classifier API can also read from an env var)
DEFAULT_MODEL_YAML = os.getenv(
    "MODEL_CONFIG_YAML",
    "backend/classification/artifacts/model_config.yaml"
)
st.session_state.setdefault("last_update", 0)
st.set_page_config(page_title="Personal Finance App", layout="wide")
page = st.sidebar.radio("Navigation", ["Upload PDF", "View Charts & Table"], key="nav")

# ------------------------------- Helpers ---------------------------------

def _ok(resp: requests.Response) -> bool:
    return (resp is not None) and resp.ok

@st.cache_data(ttl=1800)
def get_banks() -> Tuple[List[str], Dict[str, str]]:
    """
    Ask extractor_api for the supported banks (`/banks`).
    Falls back to a static list if the endpoint is not present.
    """
    try:
        r = requests.get(f"{EXTRACTOR_API_URL}/banks", timeout=5)
        if _ok(r):
            js = r.json()
            labels = [b["label"] for b in js["banks"]]
            key_by_label = {b["label"]: b["key"] for b in js["banks"]}
            return labels, key_by_label
    except Exception:
        pass
    # Fallback
    banks = [{"label": "Fineco", "key": "fineco"}, {"label": "UBS", "key": "ubs"}]
    labels = [b["label"] for b in banks]
    key_by_label = {b["label"]: b["key"] for b in banks}
    return labels, key_by_label

@st.cache_data(ttl=600)
def get_models() -> List[Dict]:
    """
    Ask classifier_api for available models (name + signature).
    Requires a small `/models` endpoint on classifier_api (see below).
    """
    r = requests.get(f"{CLASSIFIER_API_URL}/models", timeout=20)
    r.raise_for_status()
    return r.json()["models"]

def call_ingest(bank_key: str, file_name: str, content: bytes) -> Dict:
    files = {"file": (file_name, content, "application/pdf")}
    r = requests.post(f"{EXTRACTOR_API_URL}/extract/{bank_key}", files=files, timeout=180)
    r.raise_for_status()
    return r.json()

def call_classify(model_config_path: str, batch_size: int = 512) -> Dict:
    """
    Triggers classification for all models in YAML.
    The endpoint returns a summary with model_signature and how many rows were labeled now.
    """
    params = {"model_config_path": model_config_path, "batch_size": batch_size}
    r = requests.post(f"{CLASSIFIER_API_URL}/classify_transactions", params=params, timeout=300)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300)
def fetch_joined_df(model_signature: str, refresh_token: int) -> pd.DataFrame:
    """
    Pull transactions joined with predictions for a specific model signature.
    Tries JSON endpoint `/export_df`, falls back to CSV `/export.csv`.
    """
    # Prefer JSON
    try:
        r = requests.get(f"{CLASSIFIER_API_URL}/export_df", params={"model_signature": model_signature}, timeout=20)
        if _ok(r):
            rows = r.json()["rows"]
            return pd.DataFrame(rows)
    except Exception:
        pass

    # Fallback: CSV
    r = requests.get(f"{CLASSIFIER_API_URL}/export.csv", params={"model_signature": model_signature}, timeout=20)
    r.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(r.text), parse_dates=["data_operazione", "data_valuta"])


# ----------------------------- Page 1 ------------------------------------

if page == "Upload PDF":
    st.title("Upload Bank Statement PDF")

    # Bank selector
    bank_labels, bank_key_by_label = get_banks()
    bank_label = st.selectbox("Select bank", bank_labels, index=None, placeholder="Choose a bank")

    # File uploader
    uploaded = st.file_uploader("Choose a PDF", type="pdf")

    # Optionally allow a custom YAML path (defaults to env)
    with st.expander("Advanced"):
        model_yaml = st.text_input("Model config YAML path", value=DEFAULT_MODEL_YAML)
        batch_size = st.slider("Batch size", 64, 2000, 512, step=64)

    if st.button("Parse & Classify"):
        if not bank_label:
            st.warning("Please select a bank.")
            st.stop()
        if uploaded is None:
            st.warning("Please upload a PDF.")
            st.stop()

        bank_key = bank_key_by_label[bank_label]

        # (1) Ingest
        with st.spinner("Parsing PDF on backend and saving to DB..."):
            try:
                payload = call_ingest(bank_key, uploaded.name, uploaded.getvalue())
            except Exception as e:
                st.error(f"Extractor failed: {e}")
                st.stop()

        st.success(f"✅ Ingested {payload.get('ingested', 0)} rows for {bank_label}.")

        # (2) Classify (predict-only-missing per model in YAML)
        with st.spinner("Running classification (only missing rows per model)..."):
            try:
                cls_summary = call_classify(model_yaml, batch_size=batch_size)
            except Exception as e:
                st.error(f"Classifier failed: {e}")
                st.stop()

        # Show a compact summary
        st.subheader("Classification Summary")
        summary = cls_summary.get("summary", [])
        if summary:
            st.table(pd.DataFrame(summary))
        else:
            st.info("No new rows required classification (DB already up-to-date).")

        st.success("Done! Go to 'View Charts & Table' to explore.")
        st.session_state["last_update"] = int(time.time())


# ----------------------------- Page 2 ------------------------------------

else:
    st.title("Charts & Table")

    # 1) Model picker (required before showing charts)
    try:
        models = get_models()
    except Exception as e:
        st.error(f"Could not load models from classifier API: {e}")
        st.stop()

    # Build label -> signature map for UI
    if not models:
        st.warning("No models available. Run classification once from Page 1.")
        st.stop()

    model_labels = [f"{m.get('name','model')} — signature: {m['model_signature'][:8]}" for m in models]
    sig_by_label = {lbl: m["model_signature"] for lbl, m in zip(model_labels, models)}

    chosen = st.selectbox("Choose model", model_labels, index=0)
    model_sig = sig_by_label[chosen]

    # 2) Load joined data for chosen model
    with st.spinner("Loading data from DB..."):
        try:
            df = fetch_joined_df(model_sig, st.session_state["last_update"])
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

    if df.empty:
        st.info("No data yet for this model. Upload a PDF on Page 1.")
        st.stop()

    # 3) Cleanup/columns
    # Ensure dates
    for c in ("data_operazione","data_valuta"):
        if c in df.columns and not np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Ensure amount
    if "ammontare" not in df.columns:
        df["ammontare"] = 0.0
    else:
        df["ammontare"] = pd.to_numeric(df["ammontare"], errors="coerce").fillna(0.0)

    # Ensure prediction label
    pred_col = "label" if "label" in df.columns else "category"
    if pred_col not in df.columns:
        df[pred_col] = "none"

    # Month bucket
    if "data_operazione" in df.columns:
        df["month"] = df["data_operazione"].dt.to_period("M").dt.to_timestamp()
    else:
        st.warning("Column 'data_operazione' missing; charts by month may be incomplete.")
        df["month"] = pd.NaT

    # Type (sign)
    df["type"] = np.where(df["ammontare"] >= 0, "entrate", "uscite")

    # ---------------- Chart 1: Entrate vs Uscite ----------------
    left, right = st.columns((2, 1))
    with left:
        st.subheader("Entrate (↑) e Uscite (↓) Mensili")
        df_monthly_type = (
            df.groupby(["month", "type"], as_index=False)["ammontare"]
              .sum()
              .sort_values("month")
        )
        color_scale = alt.Scale(domain=["entrate", "uscite"], range=["#2ecc40", "#ff4136"])
        chart_eu = (
            alt.Chart(df_monthly_type)
            .mark_bar()
            .encode(
            x=alt.X("yearmonth(month):T", title="Month"),
            y=alt.Y("ammontare:Q", title="Amount"),
            color=alt.Color(
                "type:N",
                legend=alt.Legend(title="Tipo"),
                scale=color_scale,
            ),
            tooltip=[
                alt.Tooltip("yearmonth(month):T", title="Month"),
                alt.Tooltip("type:N", title="Tipo"),
                alt.Tooltip("ammontare:Q", title="Amount"),
            ],
            )
        )
        st.altair_chart(chart_eu, use_container_width=True)

    with right:
        st.subheader("Quick Stats")
        st.metric("Rows", len(df))
        st.metric("Total Amount", f"{df['ammontare'].sum():,.2f}")
        st.metric("# Categories", df[pred_col].nunique())

    # ---------------- Pie: Totale per Categoria ----------------
    st.subheader("Totale per Categoria")
    df_cat_total = (
        df.groupby(pred_col, as_index=False)["ammontare"]
          .sum()
          .sort_values("ammontare", ascending=False)
    )
    chart_cat_pie = (
        alt.Chart(df_cat_total)
        .mark_arc()
        .encode(
            theta=alt.Theta("ammontare:Q", title="Total Amount"),
            color=alt.Color(f"{pred_col}:N", legend=alt.Legend(title="category")),
            tooltip=[
                alt.Tooltip(f"{pred_col}:N", title="category"),
                alt.Tooltip("ammontare:Q", title="Total Amount", format=",.2f"),
            ],
        )
    )
    st.altair_chart(chart_cat_pie, use_container_width=True)

    # ---------------- Bars: Andamento Mensile per Categoria ---------------
    st.subheader("Andamento Mensile per Categoria")
    df_cat_monthly = (
        df.groupby(["month", pred_col], as_index=False)["ammontare"]
          .sum()
          .sort_values(["month", "ammontare"], ascending=[True, False])
    )
    chart_cat_monthly = (
        alt.Chart(df_cat_monthly)
        .mark_bar()
        .encode(
            x=alt.X("yearmonth(month):T", title="Month"),
            y=alt.Y("ammontare:Q", title="Amount"),
            color=alt.Color(f"{pred_col}:N", legend=alt.Legend(title="category")),
            tooltip=[
                alt.Tooltip("yearmonth(month):T", title="Month"),
                alt.Tooltip(f"{pred_col}:N", title="category"),
                alt.Tooltip("ammontare:Q", title="Amount", format=",.2f"),
            ],
        )
    )
    st.altair_chart(chart_cat_monthly, use_container_width=True)

    # ---------------- Table + Download -----------------------------------
    st.subheader("All Transactions")
    cols_show = [c for c in ["data_valuta", "ammontare", "descrizione", pred_col] if c in df.columns]
    st.dataframe(df.sort_values("data_operazione", na_position="last")[cols_show], use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name=f"transactions_with_{model_sig[:8]}.csv",
        mime="text/csv",
    )