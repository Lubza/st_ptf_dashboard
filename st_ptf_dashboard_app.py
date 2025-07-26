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

    ###
    df['settleDate'] = pd.to_datetime(df['settleDate'], format='%Y%m%d')
    df['Year'] = df['settleDate'].dt.year
    df['Month'] = df['settleDate'].dt.strftime('%b')  # Skratka mesiaca Jan, Feb, ...

    # Výber roka (dropdown)
    selected_year = st.selectbox('Vyber rok:', sorted(df['Year'].unique()))

    # Filtrovanie podľa vybraného roka
    df_year = df[df['Year'] == selected_year]

    # Agregácia podľa mesiaca a meny
    summary = (
        df_year.groupby(['Month', 'currency'])['amount']
        .sum()
        .reset_index()
    )

    # Zabezpeč zoradenie mesiacov správne:
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    summary['Month'] = pd.Categorical(summary['Month'], categories=months_order, ordered=True)
    summary = summary.sort_values('Month')

    # Vykreslenie grafu
    st.subheader(f"Summary by Month and Currency ({selected_year})")
    chart1 = alt.Chart(summary).mark_bar().encode(
        x=alt.X('Month:O', title='Month', sort=months_order),
        y=alt.Y('amount:Q', title='Sum of Dividends'),
        color=alt.Color('currency:N', title='Currency'),
        tooltip=['Month', 'currency', 'amount']
    ).properties(width=700)

    st.altair_chart(chart1, use_container_width=True)
