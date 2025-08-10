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

df_divi["amount"] = pd.to_numeric(df_divi["amount"], errors="coerce").fillna(0)

# --- jednotné mená stĺpcov v tx (kvôli jednoduchšiemu spracovaniu)
df_tx.columns = [c.lower() for c in df_tx.columns]

# Čo chceme premapovať (podľa potreby sem vieš dopĺňať ďalšie páry)
TICKER_FIX = {"VNA": "VNA.DE",
              "VNA.DIV": "VNA.DE",
              "VNA.DRTS": "VNA.DE",
              "VNA.DVD": "VNA.DE",
              "VNA.DVRTS": "VNA.DE",
              "DIC": "BRNK.DE",
              "DIC.DIV": "BRNK.DE",
              "BRNK": "BRNK.DE",
              "LI": "LI.PA",
              "RWE": "RWE.DE",
              "BYG": "BYG.L",
              "TKA": "TKA.DE",
              "TUI1": "TUI1.DE",}

# 1) Dividendy – premenovať symbol
if "symbol" in df_divi.columns:
    df_divi["symbol"] = df_divi["symbol"].replace(TICKER_FIX)

# 2) Transactions – premenovať underlying symbol
if "underlyingsymbol" in df_tx.columns:
    df_tx["underlyingsymbol"] = df_tx["underlyingsymbol"].replace(TICKER_FIX)

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

        #
        # -- len záznamy za aktuálny mesiac (podľa settledate)
        current_period = pd.Timestamp.today().to_period('M')  # napr. 2025-07
        df_divi_cur = df_divi[df_divi['settledate'].dt.to_period('M') == current_period]

        #

        # 2) Zoradíme zostupne podľa settledate
        #df_divi_sorted = df_divi.sort_values("settledate", ascending=False)
        #df_show = df_divi_sorted[["symbol","settledate_str","currency","amount"]].reset_index(drop=True)

        df_show = (
            df_divi_cur.sort_values("settledate", ascending=False)
                [["symbol", "settledate_str", "currency", "amount"]]
                .reset_index(drop=True)
        )

        # 3) Layout do dvoch stĺpcov
        col1, col2 = st.columns([2.7, 1.3])

        with col1:
            tab1, tab2, tab3 = st.tabs(
                ["📅 Rok", "🗓️ By Quarter", "🔖 Ticker"]
            )

            # ----- Tab 1: Stacked bar podľa roku a meny
            with tab1:
                # ---- graf
                chart_width = 700  # rovnaká šírka pre graf aj tabuľku
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
                #
                # === YEAR × CURRENCY (rows = currencies, columns = years) — BEZ stĺpca Total ===
                st.subheader("Year × Currency")

                width_px = 700

                summary_y = (
                    df_divi.groupby(['year', 'currency'], as_index=False)['amount']
                        .sum()
                )

                # pivot: meny v riadkoch, roky v stĺpcoch
                pivot_y = summary_y.pivot_table(
                    index='currency', columns='year', values='amount',
                    aggfunc='sum', fill_value=0
                )

                # zoradenie rokov
                year_cols = sorted(pivot_y.columns.tolist())
                pivot_y = pivot_y.reindex(columns=year_cols)

                # spodný riadok „Total“ (súčet za všetky meny) — len cez roky, BEZ stĺpca Total
                row_total = pivot_y[year_cols].sum().to_frame().T
                row_total.index = ['Total']

                pivot_y = pd.concat([pivot_y, row_total], axis=0)

                pivot_y.index.name = 'Currency'
                pivot_y.columns.name = 'Year'

                # zobrazenie
                display_df = pivot_y.reset_index()

                # číselné typy (bez ⚠️)
                num_cols = [c for c in display_df.columns if c != 'Currency']
                display_df[num_cols] = (
                    display_df[num_cols]
                    .apply(pd.to_numeric, errors='coerce')
                    .fillna(0)
                    .round(0)
                    .astype('Int64')
                )

                st.dataframe(
                    display_df,
                    width=width_px,
                    use_container_width=False,
                    hide_index=True,
                    height=min(420, 42 * (len(display_df) + 1)),
                    column_config={
                        "Currency": st.column_config.TextColumn(),
                        **{c: st.column_config.NumberColumn(format="%,d") for c in num_cols}
                    }
                )

            # ----- Tab 2: mesačný prehľad so selectbox-om zarovnaným ku grafu
            with tab2:
                st.subheader("Summary by Quarter & Currency")
                width_px = 700
                year_options = sorted(df_divi['year'].unique())
                selected_year = st.selectbox(
                    "Vyber rok:",
                    options=year_options,
                    index=len(year_options) - 1,   # => posledný = najnovší rok
                    key="sel_year"
                )
                # hack na šírku selectboxu
                st.markdown(f"""
                    <style>
                    div[data-baseweb="select"] > div {{ width: {width_px}px !important; }}
                    </style>
                """, unsafe_allow_html=True)

                # Kvartalne zobrazenie
                df_y = df_divi[df_divi['year'] == selected_year].copy()
                df_y['quarter'] = 'Q' + df_y['settledate'].dt.quarter.astype(str)

                summary_q = (
                    df_y.groupby(['quarter', 'currency'], as_index=False)['amount']
                        .sum()
                )

                order_q = ['Q1', 'Q2', 'Q3', 'Q4']
                summary_q['quarter'] = pd.Categorical(summary_q['quarter'], categories=order_q, ordered=True)
                summary_q = summary_q.sort_values('quarter')

                chart1 = alt.Chart(summary_q).mark_bar().encode(
                    x=alt.X('quarter:O', sort=order_q, title='Quarter'),
                    y=alt.Y('amount:Q', title='Sum'),
                    color=alt.Color('currency:N', title='Currency'),
                    tooltip=['quarter', 'currency', 'amount']
                ).properties(width=width_px)
                st.altair_chart(chart1, use_container_width=False)

                # --- Quarter × Currency pivot table (reacts to selected_year)

                q_order = ['Q1', 'Q2', 'Q3', 'Q4']

                pivot = (
                    summary_q
                    .assign(quarter=pd.Categorical(summary_q['quarter'], categories=q_order, ordered=True))
                    .pivot_table(index='currency', columns='quarter', values='amount', aggfunc='sum', fill_value=0)
                    .reindex(columns=q_order)
                )

                # totals
                pivot['Total'] = pivot.sum(axis=1)
                total_row = pivot.sum(axis=0).to_frame().T
                total_row.index = ['Total']
                pivot = pd.concat([pivot, total_row], axis=0)

                pivot.index.name = 'Currency'
                pivot.columns.name = 'Quarter'

                st.subheader(f"Quarter × Currency – {selected_year}")

                # -> tu je kľúčové: rovnaká šírka ako selectbox/graf a skrytý index
                display_df = pivot.reset_index()

                st.dataframe(
                    display_df,
                    width=width_px,                  # rovnaké ako pri selectboxe/grafe
                    use_container_width=False,
                    hide_index=True,                 # skryje 0/1/2/3
                    height=min(380, 42 * (len(display_df) + 1)),
                    column_config={
                        **{c: st.column_config.NumberColumn(format="%.2f") for c in q_order + ['Total']}
                    }
                )

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
                    # --- TABUĽKA pod grafom (agregácia cez všetky zvolené tickery)
                    year_totals = (
                        df_t.groupby('year', as_index=False)['amount']
                        .sum()
                    )

                    # (voliteľné) medziročná zmena – ak ju chceš, nechaj nasledujúce 4 riadky
                    tmp = year_totals.sort_values('year')        # vzostupne
                    tmp['change'] = tmp['amount'].diff().fillna(0)
                    year_totals = tmp.sort_values('year', ascending=False)

                    st.dataframe(
                        year_totals.rename(columns={'year': 'Year', 'amount': 'Total', 'change': 'Change'}),
                        use_container_width=False,
                        height=min(320, 42 * (len(year_totals) + 1)),
                        column_config={
                            "Year":  st.column_config.NumberColumn(format="%d"),
                            "Total": st.column_config.NumberColumn(format="%.2f"),
                            "Change": st.column_config.NumberColumn(format="%.2f"),
                        }
                    )
                else:
                    st.info("Vyber aspoň jeden ticker.")
        with col2:
            st.subheader("Current month dividends")
            if df_show.empty:
                st.info(" V tomto mesiaci zatial nemas ziadne dividendy")
            else:
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