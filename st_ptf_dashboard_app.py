import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os                      

DB_URL = st.secrets["DB_URL"]
TABLE_NAME = st.secrets["TABLE_NAME"]

st.title("IBKR Dividendy Dashboard")

@st.cache_data(ttl=0)
def load_data():
    engine = create_engine(DB_URL)
    return pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)

df = load_data()

if df.empty:
    st.warning("Tabuľka s dividendami je prázdna.")
else:
    st.subheader("Zoznam dividend")
    st.dataframe(df)

    st.subheader("Súhrn podľa meny")
    st.bar_chart(df.groupby("currency")["amount"].sum())

    st.subheader("Sumár za obdobie")
    df['settleDate'] = pd.to_datetime(df['settleDate'], format="%Y%m%d")
    st.line_chart(df.groupby("settleDate")["amount"].sum())