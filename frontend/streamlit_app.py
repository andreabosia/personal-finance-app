# frontend/streamlit_app.py
import time
import streamlit as st
import requests
import pandas as pd
import numpy as np
import altair as alt
import os
from typing import Dict, List, Tuple
import altair as alt
from itertools import product
import tempfile


EXTRACTOR_API_URL = os.getenv("EXTRACTOR_API_URL", "http://localhost:8000")
CLASSIFIER_API_URL = os.getenv("CLASSIFIER_API_URL", "http://localhost:8001")

# Your default YAML path (the classifier API can also read from an env var)
DEFAULT_MODEL_YAML = os.getenv(
    "MODEL_CONFIG_YAML",
    "backend/classification/artifacts/model_config.yaml"
)
st.session_state.setdefault("last_update", 0)
st.set_page_config(page_title="Personal Finance App", layout="wide")
page = st.sidebar.radio("Navigation", ["Upload PDF", "View Charts & Table", "Capital Projections"] , key="nav")

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

elif page == "View Charts & Table":
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

    # ---------------- Pie: Media mensile per Categoria ----------------
    st.subheader("Spesa Media Mensile per Categoria")


    monthly = (
        df.groupby(["month", pred_col], as_index=False)["ammontare"]
        .sum()
    )

    all_months = monthly["month"].drop_duplicates().sort_values()
    pivot = (
        monthly.pivot(index="month", columns=pred_col, values="ammontare")
            .reindex(all_months)
            .fillna(0.0)
    )

    avg_per_cat = (
        pivot.mean(axis=0)
            .reset_index()
            .rename(columns={"index": pred_col, 0: "ammontare_avg_month"})
    )
    avg_per_cat.columns = [pred_col, "ammontare_avg_month"]

    chart_cat_pie = (
        alt.Chart(avg_per_cat)
        .mark_arc()
        .encode(
            theta=alt.Theta("ammontare_avg_month:Q", title="Avg Amount per Month"),
            color=alt.Color(f"{pred_col}:N", legend=alt.Legend(title="Category")),
            tooltip=[
                alt.Tooltip(f"{pred_col}:N", title="Category"),
                alt.Tooltip("ammontare_avg_month:Q", title="Avg/Month", format=",.2f")
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

# ----------------------------- Page 3 ------------------------------------
elif page == "Capital Projections":


    st.set_page_config(page_title="Capital Projections", layout="wide")

    # ---------- Helpers ----------
    def fv_with_contributions(
        principal: float,
        contribution: float,
        years: float,
        r_annual: float,
        comp_per_year: int = 12,
    ) -> float:
        """
        Future value with regular contributions.
        principal: initial capital (P)
        contribution: contribution per contribution-period (PMT)
        years: total years (t)
        r_annual: nominal annual rate in decimal (e.g., 0.07)
        comp_per_year: compounding/contribution frequency (m)
        """
        m = comp_per_year
        r = r_annual
        i = r / m
        n = int(round(m * years))

        # If rate is ~0, avoid division by zero: FV = P + PMT*n (+ shift for begin timing)
        if abs(i) < 1e-12:
            fv = principal + contribution * n
            return fv

        # Growth of principal
        fv_principal = principal * (1 + i) ** n

        # Growth of contributions (ordinary annuity)
        fv_pmt = contribution * (((1 + i) ** n - 1) / i)

        
        fv_pmt *= (1 + i)

        return fv_principal + fv_pmt


    def apply_inflation_adjustment(series: pd.Series, inflation_annual: float, comp_per_year: int) -> pd.Series:
        """Deflate nominal values to today's money with the same compounding frequency."""
        if inflation_annual <= 0:
            return series
        i = inflation_annual / comp_per_year
        idx = np.arange(len(series))  # 0..n
        deflator = (1 + i) ** idx
        return series / deflator


    # ---------- UI ----------
    st.title("💹 Capital Projections (Interactive)")

    with st.sidebar:
        st.header("Global Settings")

        annual_return = st.number_input("Annual return (%, nominal)", min_value=-50.0, max_value=50.0, value=7.0, step=0.1) / 100.0
        comp_per_year = st.selectbox("Compounding frequency", options=[1, 2, 4, 12], index=3)
        inflation = st.number_input("Inflation (%, optional, real value output)", min_value=0.0, max_value=50.0, value=0.0, step=0.1) / 100.0

        st.markdown("---")
        st.subheader("Scenarios")
        init_capitals_str = st.text_input(
            "Initial capital (comma-separated, e.g. 13000, 50000, 100000)",
            value="13000"
        )
        try:
            init_capitals = [float(x.strip().replace("’", "").replace("_", "")) for x in init_capitals_str.split(",") if x.strip()]
        except Exception:
            st.warning("Please enter valid numbers separated by commas.")
            init_capitals = []

        monthly_savings_str = st.text_input(
            "Monthly saving (comma-separated, e.g. 500, 1000)",
            value="2000"
        )
        try:
            monthly_savings = [float(x.strip().replace("’", "").replace("_", "")) for x in monthly_savings_str.split(",") if x.strip()]
        except Exception:
            st.warning("Please enter valid numbers separated by commas.")
            monthly_savings = []

        max_years = st.slider("Projection horizon (years)", min_value=1, max_value=50, value=30, step=1)
        year_step = st.select_slider("Year step", options=[1, 2, 5, 10], value=5)

        st.markdown("---")
        show_table = st.checkbox("Show data table", value=False)

    if not init_capitals or not monthly_savings:
        st.info("Pick at least one **Initial capital** and one **Monthly saving** in the sidebar.")
        st.stop()

    years_list = list(range(0, max_years + 1, year_step))
    if years_list[0] != 0:
        years_list = [0] + years_list

    # ---------- Compute ----------
    rows = []
    for P, PMT in product(init_capitals, monthly_savings):
        nominal_values = []
        for y in years_list:
            fv = fv_with_contributions(
                principal=P,
                contribution=PMT,
                years=y,
                r_annual=annual_return,
                comp_per_year=comp_per_year
            )
            nominal_values.append(fv)

        series = pd.Series(nominal_values, index=years_list)
        real_series = apply_inflation_adjustment(series, inflation, comp_per_year)

        for y in years_list:
            rows.append({
                "Years": y,
                "Initial Capital": P,
                "Monthly Saving": PMT,
                "Projected Capital (Nominal)": series.loc[y],
                "Projected Capital (Inflation Adj.)"   : real_series.loc[y],
            })

    df = pd.DataFrame(rows)


    df["Scenario"] = (
        "Init " + df["Initial Capital"].map(lambda x: f"{int(x):,}").str.replace(",", "’") +
        " | Save " + df["Monthly Saving"].map(lambda x: f"{int(x):,}").str.replace(",", "’")
    )

    # ---------- Charts ----------

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Years:Q", title="Years"),
            y=alt.Y("Projected Capital (Nominal):Q", title="Projected Capital", axis=alt.Axis(format="~s")),
            color=alt.Color("Scenario:N", title="Scenario"),
            tooltip=[
                alt.Tooltip("Years:Q"),
                alt.Tooltip("Initial Capital:Q", format="~s"),
                alt.Tooltip("Monthly Saving:Q", format="~s"),
                alt.Tooltip("Projected Capital (Nominal):Q", format="~s")
            ],
        )
        .properties(height=520)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


    # ---------- Table ----------
    if show_table:
        st.markdown("### Data")
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ---------- Old static version (for reference) ----------
    # st.title("Capital Projections")

    # initial_capital = [13000, 1000000]#, 500000]
    # monthly_saving = [500, 1000, 2000, 3000, 4000, 10000]
    # capital_projection = []

    # for capital in initial_capital:
    #     for saving in monthly_saving:
    #         for year in range(0, 30, 5):
    #             capital_proj = (
    #                 capital * (1 + (0.07/12))**(year*12)
    #                 + saving * (((1 + (0.07/12))**(year*12) - 1) / (0.07/12))
    #             )
    #             capital_projection.append((saving, year, capital, capital_proj))

    # df_projection = pd.DataFrame(
    #     capital_projection,
    #     columns=["Monthly Saving", "Years", "Initial Capital", "Projected Capital"]
    # )

    # # add a combined label for color+stroke style
    # df_projection["Scenario"] = (
    #     "Init " + df_projection["Initial Capital"].astype(str)
    #     + " | Save " + df_projection["Monthly Saving"].astype(str)
    # )

    # chart = (
    #     alt.Chart(df_projection)
    #     .mark_line(point=True)
    #     .encode(
    #         x=alt.X("Years:O", title="Years"),
    #         y=alt.Y("Projected Capital:Q", title="Projected Capital (€)"),
    #         color=alt.Color("Scenario:N", legend=alt.Legend(title="Scenario")),
    #         strokeDash=alt.StrokeDash("Monthly Saving:Q"),  # <-- different line style too
    #         tooltip=[
    #             alt.Tooltip("Scenario:N", title="Scenario"),
    #             alt.Tooltip("Years:O", title="Years"),
    #             alt.Tooltip("Projected Capital:Q", title="Projected Capital (€)", format=",.2f"),
    #         ],
    #     )
    #     .properties(width=700, height=400, title="Capital Projection Over Time")
    # )

    # st.altair_chart(chart, use_container_width=True)
    
    # # Legend table: Monthly Saving, Years, Projected Capital
    # st.subheader("Projection Details")
    # legend_df = df_projection.copy()
    # legend_df["Projected Capital"] = legend_df["Projected Capital"].map(lambda x: f"€{x:,.2f}")
    # st.dataframe(legend_df, use_container_width=True)