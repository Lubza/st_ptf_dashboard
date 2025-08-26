import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import altair as alt
import numpy as np

st.set_page_config(layout="wide")

# === Konštanty šírok, nech je všetko konzistentné ===
CHART_WIDTH_LEFT  = 700   # graf vľavo + tabuľka pod ním (Tab 1)
CHART_WIDTH_TAB2  = 700   # selectbox/graf/tabuľka v Tab 2
CHART_WIDTH_TAB3  = 700   # graf + tabuľka v Tab 3
RIGHT_TABLE_H     = 260   # výška tabuliek vpravo

# načítame si z secrets.yml
DB_URL               = st.secrets["DB_URL"]
TABLE_DIVI           = st.secrets["TABLE_DIVI"]
TABLE_TRANSACTIONS   = st.secrets["TABLE_TRANSACTIONS"]

# --- SIDEBAR
st.sidebar.title("📂 Navigácia")
page = st.sidebar.radio(
    "Choď na stránku:",
    ("📊 Dividends Overview", "📈 Transactions", "Open option positions", "⚙️ Nastavenia"),
    key="nav"   # dôležité pri zmene zoznamu možností
)
st.sidebar.markdown("---")
# voliteľné: reset navigácie (ak by sa držal starý stav)
if st.sidebar.button("🔄 Obnoviť navigáciu"):
    st.session_state.pop("nav", None)
    st.rerun()

st.sidebar.info("Tu môžeš pridať ďalšie sekcie alebo filter.")

# --- FUNCTIONS na načítanie dát
@st.cache_data(ttl=600)
def load_dividends() -> pd.DataFrame:
    engine = create_engine(DB_URL)
    df = pd.read_sql(f"SELECT * FROM {TABLE_DIVI}", engine)
    df.columns = [c.lower() for c in df.columns]
    return df

@st.cache_data(ttl=600)
def load_transactions() -> pd.DataFrame:
    engine = create_engine(DB_URL)
    df = pd.read_sql(f"SELECT * FROM {TABLE_TRANSACTIONS}", engine)
    return df

df_divi = load_dividends()
df_tx   = load_transactions()

# čísla + základné čistenie
df_divi["amount"] = pd.to_numeric(df_divi["amount"], errors="coerce")
df_divi.replace([np.inf, -np.inf], np.nan, inplace=True)
df_divi["amount"] = df_divi["amount"].fillna(0)

# --- jednotné mená stĺpcov v tx (kvôli jednoduchšiemu spracovaniu)
df_tx.columns = [c.lower() for c in df_tx.columns]

# mapovanie tickerov
TICKER_FIX = {
    "VNA": "VNA.DE", "VNA.DIV": "VNA.DE", "VNA.DRTS": "VNA.DE", "VNA.DVD": "VNA.DE", "VNA.DVRTS": "VNA.DE",
    "DIC": "BRNK.DE", "DIC.DIV": "BRNK.DE", "BRNK": "BRNK.DE",
    "LI": "LI.PA", "RWE": "RWE.DE", "BYG": "BYG.L", "TKA": "TKA.DE", "TUI1": "TUI1.DE",
}
if "symbol" in df_divi.columns:
    df_divi["symbol"] = df_divi["symbol"].replace(TICKER_FIX)
if "underlyingsymbol" in df_tx.columns:
    df_tx["underlyingsymbol"] = df_tx["underlyingsymbol"].replace(TICKER_FIX)

# --- STRÁNKA: Dividends Overview
if page == "📊 Dividends Overview":
    st.title("Dividends overview")

    if df_divi.empty:
        st.warning("The dividends table is empty.")
    else:
        # dátumy
        df_divi['settledate'] = pd.to_datetime(df_divi['settledate'], format='%Y%m%d')
        df_divi['month']      = df_divi['settledate'].dt.strftime('%b')
        df_divi['year']       = df_divi['settledate'].dt.year
        df_divi['settledate_str'] = df_divi['settledate'].dt.strftime('%m/%d/%Y')

        # aktuálny mesiac (na pravý stĺpec)
        current_period = pd.Timestamp.today().to_period('M')
        df_divi_cur = df_divi[df_divi['settledate'].dt.to_period('M') == current_period]

        df_show = (
            df_divi_cur.sort_values("settledate", ascending=False)
            [["symbol", "settledate_str", "currency", "amount"]]
            .reset_index(drop=True)
        )

        # layout
        col1, col2 = st.columns([2.7, 1.3])

        # ======================= ĽAVÝ STĹPEC =======================
        with col1:
            tab1, tab2, tab3 = st.tabs(["📅 Rok", "🗓️ By Quarter", "🔖 Ticker"])

            # ----- Tab 1: Rok
            with tab1:
                st.subheader("Summary by Year & Currency")
                summary = df_divi.groupby(['year','currency'])['amount'].sum().reset_index()

                chart = (
                    alt.Chart(summary)
                    .mark_bar()
                    .encode(
                        x=alt.X('year:O', title='Year'),
                        y=alt.Y('amount:Q', title='Sum of Dividends'),
                        color=alt.Color('currency:N', title='Currency'),
                        tooltip=['year','currency','amount']
                    )
                    .properties(width=CHART_WIDTH_LEFT)
                )
                st.altair_chart(chart, use_container_width=False)

                st.subheader("Year × Currency")

                # pivot
                summary_y = df_divi.groupby(['year', 'currency'], as_index=False)['amount'].sum()
                pivot_y = summary_y.pivot_table(index='currency', columns='year', values='amount',
                                                aggfunc='sum', fill_value=0)

                year_cols = sorted(pivot_y.columns.tolist())
                pivot_y = pivot_y.reindex(columns=year_cols)

                row_total = pivot_y[year_cols].sum().to_frame().T
                row_total.index = ['Total']
                pivot_y = pd.concat([pivot_y, row_total], axis=0)

                pivot_y.index.name = 'Currency'
                pivot_y.columns.name = 'Year'
                display_df = pivot_y.reset_index()

                num_cols = [c for c in display_df.columns if c != 'Currency']
                display_df[num_cols] = (
                    display_df[num_cols]
                    .apply(pd.to_numeric, errors='coerce')
                    .fillna(0)
                    .round(0)
                    .astype(int)
                )

                st.dataframe(
                    display_df,
                    width=CHART_WIDTH_LEFT,
                    use_container_width=False,
                    hide_index=True,
                    height=min(420, 42 * (len(display_df) + 1)),
                    column_config={
                        "Currency": st.column_config.TextColumn(),
                        **{c: st.column_config.NumberColumn(format="%,d") for c in num_cols}
                    }
                )

            # ----- Tab 2: Quarter
            with tab2:
                st.subheader("Summary by Quarter & Currency")
                year_options = sorted(df_divi['year'].unique())
                selected_year = st.selectbox(
                    "Vyber rok:",
                    options=year_options,
                    index=len(year_options) - 1,
                    key="sel_year"
                )
                st.markdown(f"""
                    <style>
                    div[data-baseweb="select"] > div {{ width: {CHART_WIDTH_TAB2}px !important; }}
                    </style>
                """, unsafe_allow_html=True)

                df_y = df_divi[df_divi['year'] == selected_year].copy()
                df_y['quarter'] = 'Q' + df_y['settledate'].dt.quarter.astype(str)

                summary_q = df_y.groupby(['quarter', 'currency'], as_index=False)['amount'].sum()

                order_q = ['Q1', 'Q2', 'Q3', 'Q4']
                summary_q['quarter'] = pd.Categorical(summary_q['quarter'], categories=order_q, ordered=True)
                summary_q = summary_q.sort_values('quarter')

                chart1 = (
                    alt.Chart(summary_q)
                    .mark_bar()
                    .encode(
                        x=alt.X('quarter:O', sort=order_q, title='Quarter'),
                        y=alt.Y('amount:Q', title='Sum'),
                        color=alt.Color('currency:N', title='Currency'),
                        tooltip=['quarter', 'currency', 'amount']
                    )
                    .properties(width=CHART_WIDTH_TAB2)
                )
                st.altair_chart(chart1, use_container_width=False)

                pivot = (
                    summary_q
                    .assign(quarter=pd.Categorical(summary_q['quarter'], categories=order_q, ordered=True))
                    .pivot_table(index='currency', columns='quarter', values='amount', aggfunc='sum', fill_value=0)
                    .reindex(columns=order_q)
                )

                pivot['Total'] = pivot.sum(axis=1)
                total_row = pivot.sum(axis=0).to_frame().T
                total_row.index = ['Total']
                pivot = pd.concat([pivot, total_row], axis=0)

                pivot.index.name = 'Currency'
                pivot.columns.name = 'Quarter'

                st.subheader(f"Quarter × Currency – {selected_year}")
                display_df = pivot.reset_index()

                st.dataframe(
                    display_df,
                    width=CHART_WIDTH_TAB2,
                    use_container_width=False,
                    hide_index=True,
                    height=min(380, 42 * (len(display_df) + 1)),
                    column_config={
                        **{c: st.column_config.NumberColumn(format="%.2f") for c in order_q + ['Total']}
                    }
                )

            # ----- Tab 3: Ticker
            with tab3:
                st.subheader("Summary by Ticker & Year")
                tics = sorted(df_divi['symbol'].dropna().unique())
                sel_t = st.multiselect("Zvoľ ticker(y):", options=tics, default=tics[:1], key="sel_t")

                if sel_t:
                    df_t = df_divi[df_divi['symbol'].isin(sel_t)]
                    summ_t = df_t.groupby(['year','symbol'])['amount'].sum().reset_index()

                    chart2 = (
                        alt.Chart(summ_t)
                        .mark_bar()
                        .encode(
                            x=alt.X('year:O', title='Year'),
                            y=alt.Y('amount:Q'),
                            color=alt.Color('symbol:N', title='Ticker'),
                            tooltip=['year','symbol','amount']
                        )
                        .properties(width=CHART_WIDTH_TAB3)
                    )
                    st.altair_chart(chart2, use_container_width=False)

                    year_totals = df_t.groupby('year', as_index=False)['amount'].sum()
                    tmp = year_totals.sort_values('year')
                    tmp['change'] = tmp['amount'].diff().fillna(0)
                    year_totals = tmp.sort_values('year', ascending=False)

                    st.dataframe(
                        year_totals.rename(columns={'year': 'Year', 'amount': 'Total', 'change': 'Change'}),
                        width=CHART_WIDTH_TAB3,
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

        # ======================= PRAVÝ STĹPEC =======================
        with col2:
            st.subheader("Current month dividends")
            if df_show.empty:
                st.info(" V tomto mesiaci zatiaľ nemáš žiadne dividendy")
            else:
                st.dataframe(df_show.set_index("symbol"), height=RIGHT_TABLE_H, use_container_width=True)

            st.divider()
            st.subheader("Top 5 dividends by ticker (All-time)")

            if df_divi.empty:
                st.info("Zatiaľ tu nemáš žiadne dividendy.")
            else:
                df = df_divi[["symbol", "amount"]].copy()
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(subset=["symbol", "amount"], inplace=True)
                df["symbol"] = df["symbol"].astype(str).str.strip()

                all_time = (
                    df.groupby("symbol", as_index=False)
                    .agg(Total=("amount", "sum"))
                    .sort_values("Total", ascending=False)
                    .head(5)
                    .rename(columns={"symbol": "Ticker"})
                    .reset_index(drop=True)
                )

                all_time["Total"] = all_time["Total"].astype(float)

                st.dataframe(
                    all_time,
                    use_container_width=True,
                    hide_index=True,
                    height=RIGHT_TABLE_H,
                    column_config={
                        "Ticker": st.column_config.TextColumn(),
                        "Total":  st.column_config.NumberColumn(format="%.0f"),
                    },
                )

# --- STRÁNKA: Transactions
elif page == "📈 Transactions":
    st.header("Transactions overview")
    if df_tx.empty:
        st.warning("No transactions in the table.")
    else:
        st.dataframe(df_tx)

# --- STRÁNKA: Open option positions (len net-otvorené opcie)
elif page == "Open option positions":
    st.header("Open option positions")

    if df_tx.empty:
        st.warning("No transactions in the table.")
    else:
        required = {"assetclass", "description", "quantity"}
        missing = required - set(df_tx.columns)
        if missing:
            st.error(f"Missing columns in transactions table: {', '.join(sorted(missing))}")
        else:
            df_opt = df_tx.copy()

            # len opcie
            df_opt["assetclass"] = df_opt["assetclass"].astype(str).str.upper()
            df_opt = df_opt[df_opt["assetclass"] == "OPT"]

            # numerická quantity
            df_opt["quantity"] = pd.to_numeric(df_opt["quantity"], errors="coerce").fillna(0.0)

            # agregácia na symbol
            open_pos = (
                df_opt.groupby("description", as_index=False)["quantity"]
                .sum()
                .rename(columns={"quantity": "net_quantity"})
            )

            # odstrániť čistú nulu (s toleranciou)
            open_pos = open_pos[open_pos["net_quantity"].round(8) != 0]

            # voliteľné meta stĺpce (prvý nenull záznam na symbol)
            possible_meta = ["underlyingsymbol", "put/call", "buy/sell", "quantity", "ibcommissioncurrency", "tradeprice", "expiry", "strike", "tradedate"]
            meta_cols = [c for c in possible_meta if c in df_opt.columns]
            if meta_cols:
                meta = (
                    df_opt.groupby("description", as_index=False)[meta_cols]
                    .agg(lambda s: s.dropna().iloc[0] if s.dropna().size else None)
                )
                open_pos = open_pos.merge(meta, on="description", how="left")

            # zoradiť podľa absolútnej veľkosti
            open_pos = open_pos.sort_values("net_quantity", key=lambda s: s.abs(), ascending=False)

            if open_pos.empty:
                st.success("Žiadne otvorené opčné pozície (netto = 0).")
            else:
                st.dataframe(
                    open_pos,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "description": st.column_config.TextColumn("description"),
                        "net_quantity": st.column_config.NumberColumn("Net quantity", format="%.2f"),
                    },
                )

            # --------- SHOW DETAILS ----------
            st.markdown("---")
            show_details = st.checkbox("Show details")
            if show_details and not open_pos.empty:
                sel = st.selectbox("description:", options=open_pos["description"].tolist())
                details = df_opt[df_opt["description"] == sel].copy()

                # nájdi vhodný dátumový stĺpec a zoraď
                date_candidates = [c for c in ("date", "tradedate", "settledate", "lasttradingday") if c in details.columns]
                if date_candidates:
                    sort_col = date_candidates[0]
                    with pd.option_context('mode.chained_assignment', None):
                        try:
                            details[sort_col] = pd.to_datetime(details[sort_col], errors="coerce")
                        except Exception:
                            pass
                    details = details.sort_values(sort_col, ascending=False)

                prefer = [
                    "description","underlyingsymbol","right","expiry","strike",
                    "quantity","price","amount","transactiontype","currency","multiplier"
                ]
                cols = [c for c in prefer if c in details.columns]
                # pridaj dátumový stĺpec na začiatok, ak sme ho našli
                if date_candidates:
                    cols = [date_candidates[0]] + [c for c in cols if c != date_candidates[0]]

                st.dataframe(details[cols] if cols else details, use_container_width=True, hide_index=True)

# --- STRÁNKA: Nastavenia
else:
    st.header("Nastavenia")
    st.info("Tu budú konfiguračné možnosti aplikácie.")
