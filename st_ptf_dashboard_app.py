import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import altair as alt

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

    # Na datetime (pre ďalšie použitie)
    df['settleDate'] = pd.to_datetime(df['settleDate'], format='%Y%m%d')

    df['Month'] = df['settleDate'].dt.strftime('%b')
    df['Year'] = df['settleDate'].dt.year
    df['settleDate_str'] = df['settleDate'].dt.strftime('%m/%d/%Y')

    st.dataframe(df)

    st.subheader("Summary by Year (in original currency)")
    st.bar_chart(df.groupby("Year")["amount"].sum())

    # Agreguj podľa rok a mena
    summary = df.groupby(['Year', 'currency'])['amount'].sum().reset_index()

    st.subheader("Summary by Year and Currency (Stacked Bar Chart)")
    chart = alt.Chart(summary).mark_bar().encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('amount:Q', title='Sum of Dividends'),
        color=alt.Color('currency:N', title='Currency'),
        tooltip=['Year', 'currency', 'amount']
    ).properties(width=600)

    st.altair_chart(chart, use_container_width=True)
