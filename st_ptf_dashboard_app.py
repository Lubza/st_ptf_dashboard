import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os                      

DB_URL = st.secrets["DB_URL"]
TABLE_NAME = st.secrets["TABLE_NAME"]

st.title("Dividends overview")

@st.cache_data(ttl=0)
def load_data():
    engine = create_engine(DB_URL)
    return pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)

df = load_data()

if df.empty:
    st.warning("The dividends table is empty.")
else:
    st.subheader("List of Dividends")

    # First convert to datetime (keep datetime object for extraction)
    df['settleDate'] = pd.to_datetime(df['settleDate'], format='%Y%m%d')

    # Extract month as abbreviation (Jan, Feb, ...)
    df['Month'] = df['settleDate'].dt.strftime('%b')

    # Extract year as number
    df['Year'] = df['settleDate'].dt.year

    # Then reformat date for display (e.g., in Streamlit)
    df['settleDate'] = df['settleDate'].dt.strftime('%m/%d/%Y')
    
    st.dataframe(df)

    st.subheader("Summary by Year")
    st.bar_chart(df.groupby("Year")["amount"].sum())

    st.subheader("Summary over Time")
    df['settleDate'] = pd.to_datetime(df['settleDate'], format="%Y%m%d")
    st.line_chart(df.groupby("settleDate")["amount"].sum())