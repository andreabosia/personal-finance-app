# import streamlit as st 
# import json
# import requests
# st.title("Basic Caculator App")
# # taking user inpputs
# option = st.selectbox( 'What operation You want to perform?',
#                     ('Addition', 'Subtraction', 'Multiplication', 'Division'))
# st.write("")
# st.write("Select the numbers from slider below")
# x = st.slider("X", 0, 100, 20)
# y = st.slider("Y", 0, 130, 10)

# #converting the inputs into a json format
# inputs = {"operation": option, "x": x, "y": y}
# # When the user clicks on button it will fetch the API
# if st.button('Calculate'):
#     res = requests.post(url = "http://127.0.0.1:8000/calculate", data = json.dumps(inputs))
#     st. subheader (f"Response from API & = {res.text}")


import streamlit as st
import requests
import pandas as pd
import altair as alt
import os

BACKEND_URL = "http://localhost:8000"
CSV_PATH = "data/transactions.csv"

st.set_page_config(page_title="Personal Finance App", layout="wide")

# --- Sidebar navigation ---
page = st.sidebar.radio("Navigation", ["Upload PDF", "View Charts & Table"])

# --- Page 1: Upload PDF ---
if page == "Upload PDF":
    st.title("Upload Bank Statement PDF")

    uploaded = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded is not None:
        with st.spinner("Parsing PDF on backend..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            resp = requests.post(f"{BACKEND_URL}/extract", files=files)

        if not resp.ok:
            st.error(f"Backend request failed (HTTP {resp.status_code}).")
        else:
            payload = {}
            try:
                payload = resp.json()
            except Exception:
                st.error(f"Backend did not return JSON:\n{resp.text[:500]}")
                st.stop()

            if payload.get("ok"):
                st.success(
                    f"✅ Added {payload.get('rows_added', 0)} rows. "
                    f"Total rows: {payload.get('total_rows', 0)}.\n\n"
                    f"Saved to: {payload.get('csv_path')}"
                )
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