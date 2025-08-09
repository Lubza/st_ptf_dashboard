import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import altair as alt

st.set_page_config(layout="wide")

# načítame si z secrets.yml
DB_URL               = st.secrets["DB_URL"]
TABLE_DIVI           = st.secrets["TABLE_DIVI"]
TABLE_TRANSACTIONS   = st.secrets["TABLE_TRANSACTIONS"]

# --- SIDEBAR
st.sidebar.title("📂 Navigácia")
page = st.sidebar.radio(
    "Choď na stránku:",
    ["📊 Dividends Overview", "📈 Transactions", "⚙️ Nastavenia"]
)
st.sidebar.markdown("---")
st.sidebar.info("Tu môžeš pridať ďalšie sekcie alebo filter.")

# --- FUNCTIONS na načítanie dát
@st.cache_data(ttl=600)
def load_dividends() -> pd.DataFrame:
    engine = create_engine(DB_URL)
    df = pd.read_sql(f"SELECT * FROM {TABLE_DIVI}", engine)
    # všetky názvy stĺpcov lowercase (Postgres ich tak vyexportuje)
    df.columns = [c.lower() for c in df.columns]
    return df

@st.cache_data(ttl=600)
def load_transactions() -> pd.DataFrame:
    engine = create_engine(DB_URL)
    df = pd.read_sql(f"SELECT * FROM {TABLE_TRANSACTIONS}", engine)
    # ak treba, tiež by si mohol stĺpce lowercasovať
    return df

df_divi = load_dividends()
df_tx   = load_transactions()

# --- STRÁNKA: Dividends Overview
if page == "📊 Dividends Overview":
    st.title("Dividends overview")

    if df_divi.empty:
        st.warning("The dividends table is empty.")
    else:
        # 1) Parsujeme dátum pod správnym menom
        #    po lowercasovaní ho máme v df_divi stĺpec 'settledate'
        df_divi['settledate'] = pd.to_datetime(df_divi['settledate'], format='%Y%m%d')
        df_divi['month']      = df_divi['settledate'].dt.strftime('%b')
        df_divi['year']       = df_divi['settledate'].dt.year
        df_divi['settledate_str'] = df_divi['settledate'].dt.strftime('%m/%d/%Y')

        # 2) Zoradíme zostupne podľa settledate
        df_divi_sorted = df_divi.sort_values("settledate", ascending=False)
        df_show = df_divi_sorted[["symbol","settledate_str","currency","amount"]].reset_index(drop=True)

        # 3) Layout do dvoch stĺpcov
        col1, col2 = st.columns([2.7, 1.3])

        with col1:
            tab1, tab2, tab3 = st.tabs(
                ["📅 Rok", "🗓️ Mesiac", "🔖 Ticker"]
            )

            # ----- Tab 1: Stacked bar podľa roku a meny
            with tab1:
                summary = (
                    df_divi.groupby(['year','currency'])['amount']
                    .sum().reset_index()
                )
                st.subheader("Summary by Year & Currency")
                chart = alt.Chart(summary).mark_bar().encode(
                    x=alt.X('year:O', title='Year'),
                    y=alt.Y('amount:Q', title='Sum of Dividends'),
                    color=alt.Color('currency:N', title='Currency'),
                    tooltip=['year','currency','amount']
                ).properties(width=600)
                st.altair_chart(chart, use_container_width=False)

            # ----- Tab 2: mesačný prehľad so selectbox-om zarovnaným ku grafu
            with tab2:
                st.subheader("Summary by Month & Currency")
                width_px = 700
                selected_year = st.selectbox(
                    "Vyber rok:",
                    options=sorted(df_divi['year'].unique()),
                    key="sel_year"
                )
                # hack na šírku selectboxu
                st.markdown(f"""
                    <style>
                    div[data-baseweb="select"] > div {{ width: {width_px}px !important; }}
                    </style>
                """, unsafe_allow_html=True)

                df_y = df_divi[df_divi['year']==selected_year]
                summary_m = (
                    df_y.groupby(['month','currency'])['amount']
                    .sum().reset_index()
                )
                order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
                summary_m['month'] = pd.Categorical(summary_m['month'], categories=order, ordered=True)
                summary_m = summary_m.sort_values('month')

                chart1 = alt.Chart(summary_m).mark_bar().encode(
                    x=alt.X('month:O', sort=order, title='Month'),
                    y=alt.Y('amount:Q', title='Sum'),
                    color=alt.Color('currency:N'),
                    tooltip=['month','currency','amount']
                ).properties(width=width_px)
                st.altair_chart(chart1, use_container_width=False)

            # ----- Tab 3: výber tickerov (multiselect)
            with tab3:
                st.subheader("Summary by Ticker & Year")
                tics = sorted(df_divi['symbol'].dropna().unique())
                sel_t = st.multiselect("Zvoľ ticker(y):", options=tics, default=tics[:1], key="sel_t")
                if sel_t:
                    df_t = df_divi[df_divi['symbol'].isin(sel_t)]
                    summ_t = df_t.groupby(['year','symbol'])['amount'].sum().reset_index()
                    chart2 = alt.Chart(summ_t).mark_bar().encode(
                        x=alt.X('year:O', title='Year'),
                        y=alt.Y('amount:Q'),
                        color=alt.Color('symbol:N', title='Ticker'),
                        tooltip=['year','symbol','amount']
                    ).properties(width=600)
                    st.altair_chart(chart2, use_container_width=False)
                else:
                    st.info("Vyber aspoň jeden ticker.")
        with col2:
            st.dataframe(df_show, height=300)

# --- STRÁNKA: Transactions
elif page == "📈 Transactions":
    st.header("Transactions overview")
    if df_tx.empty:
        st.warning("No transactions in the table.")
    else:
        st.dataframe(df_tx)

# --- STRÁNKA: Nastavenia
else:
    st.header("Nastavenia")
    st.info("Tu budú konfiguračné možnosti aplikácie.")
