import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import altair as alt

st.sidebar.error("🚨 Toto je NOVÁ verze – pokud to nevidíš, běží stará verze!")

st.set_page_config(layout="wide")  # stránka bude širšia

DB_URL = st.secrets["DB_URL"]
TABLE_DIVI = st.secrets["TABLE_DIVI"]

# --- SIDEBAR (pravý panel)
st.sidebar.title("📂 Navigácia")
page = st.sidebar.radio(
    "Choď na stránku:",
    ["📊 Dividends Overview", "📈 Analýza", "⚙️ Nastavenia"]
)

st.sidebar.markdown("---")
st.sidebar.info("Tu môžeš pridať ďalšie sekcie alebo filter.")

# --- HLAVNÝ OBSAH
st.title("Dividends overview")

@st.cache_data(ttl=0)
def load_data():
    engine = create_engine(DB_URL)
    return pd.read_sql(f"SELECT * FROM {TABLE_DIVI}", engine)

df = load_data()

if page == "📊 Dividends Overview":  # 🔹 HLAVNÁ STRÁNKA
    if df.empty:
        st.warning("The dividends table is empty.")
    else:
        df['settleDate'] = pd.to_datetime(df['settleDate'], format='%Y%m%d')
        df['Month'] = df['settleDate'].dt.strftime('%b')
        df['Year'] = df['settleDate'].dt.year
        df['settleDate_str'] = df['settleDate'].dt.strftime('%m/%d/%Y')

        df_sorted = df.sort_values("settleDate", ascending=False)
        df_show = df_sorted[["symbol", "settleDate_str", "currency", "amount"]].reset_index(drop=True)

        # --- Rozloženie do stĺpcov
        col1, col2 = st.columns([1.3, 2.7])  # Pomer šírok namiesto 1.3, 2.7

        with col1:
            st.dataframe(df_show, height=200)

        with col2:
            tab1, tab2, tab3 = st.tabs(
                ["Súhrn podľa roka", "Súhrn podľa mesiaca", "Súhrn podľa tickera"]
            )

            with tab1:
                summary = df.groupby(['Year', 'currency'])['amount'].sum().reset_index()
                st.subheader("Summary by Year and Currency (Stacked Bar Chart)")
                chart = alt.Chart(summary).mark_bar().encode(
                    x=alt.X('Year:O', title='Year'),
                    y=alt.Y('amount:Q', title='Sum of Dividends'),
                    color=alt.Color('currency:N', title='Currency'),
                    tooltip=['Year', 'currency', 'amount']
                ).properties(width=600)
                st.altair_chart(chart, use_container_width=False)

            with tab2:
                st.subheader("Mesačný prehľad podľa roka")
                width_px = 700

                selected_year = st.selectbox(
                    'Vyber rok:',
                    sorted(df['Year'].unique()),
                    key="year_select"
                )
                st.markdown(
                    f"""<style>
                    div[data-baseweb="select"] > div {{
                        width: {width_px}px !important;
                    }}
                    </style>""",
                    unsafe_allow_html=True,
                )

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
                ).properties(width=width_px)

                st.altair_chart(chart1, use_container_width=False)

            with tab3:
                st.subheader("Výber tickerov")
                ticker_options = sorted(df['symbol'].dropna().unique())
                selected_tickers = st.multiselect(
                    "Zvoľ jeden alebo viac tickerov:",
                    options=ticker_options,
                    default=ticker_options[:1],
                    key="ticker_select"
                )

                if selected_tickers:
                    df_ticker = df[df['symbol'].isin(selected_tickers)]
                    summary_ticker = (
                        df_ticker.groupby(['Year', 'symbol'])['amount']
                        .sum()
                        .reset_index()
                    )
                    st.subheader(f"Súhrn dividend podľa tickera a roka")
                    chart2 = alt.Chart(summary_ticker).mark_bar().encode(
                        x=alt.X('Year:O', title='Year'),
                        y=alt.Y('amount:Q', title='Sum of Dividends'),
                        color=alt.Color('symbol:N', title='Ticker'),
                        tooltip=['Year', 'symbol', 'amount']
                    ).properties(width=600)
                    st.altair_chart(chart2, use_container_width=False)
                else:
                    st.info("Vyber aspoň jeden ticker na zobrazenie grafu.")

elif page == "📈 Transactions":
    st.header("📈 Transactions")
    st.info("Tu môžeš neskôr doplniť grafy pre podrobnejšiu analýzu.")

elif page == "⚙️ Nastavenia":
    st.header("⚙️ Nastavenia")
    st.info("Tu budú konfiguračné možnosti aplikácie.")
