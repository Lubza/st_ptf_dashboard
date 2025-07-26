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
    # Pre istotu pretypuj dátum len raz
    df['settleDate'] = pd.to_datetime(df['settleDate'], format='%Y%m%d')
    df['Month'] = df['settleDate'].dt.strftime('%b')
    df['Year'] = df['settleDate'].dt.year
    df['settleDate_str'] = df['settleDate'].dt.strftime('%m/%d/%Y')

    st.dataframe(df)

    tab1, tab2 = st.tabs(["Súhrn podľa roka", "Súhrn podľa mesiaca"])

    with tab1:
        st.subheader("Summary by Year (in original currency)")
        st.bar_chart(df.groupby("Year")["amount"].sum())

        summary = df.groupby(['Year', 'currency'])['amount'].sum().reset_index()
        st.subheader("Summary by Year and Currency (Stacked Bar Chart)")
        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('amount:Q', title='Sum of Dividends'),
            color=alt.Color('currency:N', title='Currency'),
            tooltip=['Year', 'currency', 'amount']
        ).properties(width=600)
        st.altair_chart(chart, use_container_width=True)

    with tab2:
        st.subheader("Mesačný prehľad podľa roka")

        selected_year = st.selectbox('Vyber rok:', sorted(df['Year'].unique()), key="year_select")
        df_year = df[df['Year'] == selected_year]
        summary_month = (
            df_year.groupby(['Month', 'currency'])['amount']
            .sum()
            .reset_index()
        )
        months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        summary_month['Month'] = pd.Categorical(summary_month['Month'], categories=months_order, ordered=True)
        summary_month = summary_month.sort_values('Month')
        st.subheader(f"Summary by Month and Currency ({selected_year})")
        chart1 = alt.Chart(summary_month).mark_bar().encode(
            x=alt.X('Month:O', title='Month', sort=months_order),
            y=alt.Y('amount:Q', title='Sum of Dividends'),
            color=alt.Color('currency:N', title='Currency'),
            tooltip=['Month', 'currency', 'amount']
        ).properties(width=700)
        st.altair_chart(chart1, use_container_width=True)
