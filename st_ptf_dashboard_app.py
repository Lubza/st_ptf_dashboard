import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import altair as alt
import numpy as np
import yfinance as yf

st.set_page_config(layout="wide")

# === Layout constants ===
CHART_WIDTH_LEFT  = 700
CHART_WIDTH_TAB2  = 700
CHART_WIDTH_TAB3  = 700
RIGHT_TABLE_H     = 260

# --- Secrets
DB_URL               = st.secrets["DB_URL"]
TABLE_DIVI           = st.secrets["TABLE_DIVI"]
TABLE_TRANSACTIONS   = st.secrets["TABLE_TRANSACTIONS"]

# --- SIDEBAR
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ("📊 Dividends Overview", "📈 Transactions", "Open option positions", "Open stock positions", "⚙️ Settings"),
    key="nav"
)
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset navigation"):
    st.session_state.pop("nav", None)
    st.rerun()
st.sidebar.info("You can add more sections or filters here.")

# --- DATA LOADERS
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
    df.columns = [c.lower() for c in df.columns]
    return df

df_divi = load_dividends()
df_tx   = load_transactions()

@st.cache_data(ttl=3600)
def fetch_eod_close(symbols: list[str]) -> pd.DataFrame:
    """
    Fetch latest available EOD Close for each Yahoo ticker in `symbols`.
    - For BYG.L convert GBp -> GBP (divide by 100) right after fetch.
    Returns DataFrame: ['Symbol', 'Current price'].
    """
    if yf is None:
        return pd.DataFrame({"Symbol": symbols, "Current price": [np.nan]*len(symbols)})

    tickers = [s for s in dict.fromkeys([str(s).strip() for s in symbols]) if s]
    if not tickers:
        return pd.DataFrame(columns=["Symbol", "Current price"])

    data = yf.download(
        tickers=" ".join(tickers),
        period="10d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    div_by_100_upper = {"BYG.L"}  # LSE tickers quoted in GBp

    closes = []
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):  # multiple tickers
                s = data[(t, "Close")].dropna()
            else:                                        # single ticker
                s = data["Close"].dropna()
            price = float(s.iloc[-1]) if len(s) else np.nan
        except Exception:
            price = np.nan

        if t.upper() in div_by_100_upper and pd.notna(price):
            price = price / 100.0

        closes.append({"Symbol": t, "Current price": price})

    return pd.DataFrame(closes)

# --- Basic cleanup for dividends
if not df_divi.empty:
    df_divi["amount"] = pd.to_numeric(df_divi.get("amount", 0), errors="coerce")
    df_divi.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_divi["amount"] = df_divi["amount"].fillna(0)

# ========================= PAGE: Dividends Overview =========================
if page == "📊 Dividends Overview":
    st.title("Dividends overview")

    if df_divi.empty:
        st.warning("The dividends table is empty.")
    else:
        # Dates & helpers
        df_divi['settledate'] = pd.to_datetime(df_divi['settledate'], format='%Y%m%d', errors='coerce')
        df_divi['month']      = df_divi['settledate'].dt.strftime('%b')
        df_divi['year']       = df_divi['settledate'].dt.year
        df_divi['settledate_str'] = df_divi['settledate'].dt.strftime('%m/%d/%Y')

        current_period = pd.Timestamp.today().to_period('M')
        df_divi_cur = df_divi[df_divi['settledate'].dt.to_period('M') == current_period]

        df_show = (
            df_divi_cur.sort_values("settledate", ascending=False)
            [["symbol", "settledate_str", "currency", "amount"]]
            .reset_index(drop=True)
        )

        col1, col2 = st.columns([2.7, 1.3])

        # ----------------------- LEFT COLUMN -----------------------
        with col1:
            tab1, tab2, tab3 = st.tabs(["📅 Year", "🗓️ By Quarter", "🔖 Ticker"])

            # ----- Tab 1: Year
            with tab1:
                st.subheader("Summary by Year & Currency")
                summary = df_divi.groupby(['year','currency'], dropna=False)['amount'].sum().reset_index()

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
                year_options = sorted(df_divi['year'].dropna().unique())
                selected_year = st.selectbox(
                    "Select year:",
                    options=year_options,
                    index=len(year_options) - 1 if len(year_options) else 0,
                    key="sel_year"
                )
                st.markdown(f"""
                    <style>
                    div[data-baseweb="select"] > div {{ width: {CHART_WIDTH_TAB2}px !important; }}
                    </style>
                """, unsafe_allow_html=True)

                df_y = df_divi[df_divi['year'] == selected_year].copy()
                df_y['quarter'] = 'Q' + df_y['settledate'].dt.quarter.astype(str)

                summary_q = df_y.groupby(['quarter', 'currency'], as_index=False, dropna=False)['amount'].sum()
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
                sel_t = st.multiselect("Choose ticker(s):", options=tics, default=tics[:1], key="sel_t")

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
                    st.info("Select at least one ticker.")

        # ----------------------- RIGHT COLUMN -----------------------
        with col2:
            st.subheader("Current month dividends")
            if df_show.empty:
                st.info("You have no dividends this month yet.")
            else:
                st.dataframe(df_show.set_index("symbol"), height=RIGHT_TABLE_H, use_container_width=True)

            st.divider()
            st.subheader("Top 5 dividends by ticker (All-time)")

            if df_divi.empty:
                st.info("No dividends yet.")
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

        # --- Divider + EXPANDER: all dividend transactions (all-time)
        st.markdown("---")
        with st.expander("Show all dividend transactions (all time)"):
            if df_divi.empty:
                st.info("No dividends in the table.")
            else:
                d = df_divi.copy()
                # date parse + sort
                if "settledate" in d.columns:
                    d["settledate"] = pd.to_datetime(d["settledate"], format="%Y%m%d", errors="coerce")
                    date_col = "settledate"
                else:
                    date_col = next((c for c in ["paymentdate", "date"] if c in d.columns), None)
                    if date_col:
                        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")

                cur_col = "currency" if "currency" in d.columns else ("currencyprimary" if "currencyprimary" in d.columns else None)

                base_cols = ["symbol"] + (["description"] if "description" in d.columns else []) + [date_col] + ([cur_col] if cur_col else []) + ["amount"]
                display_cols = [c for c in base_cols if c]

                d_disp = d[display_cols].copy().sort_values(date_col, ascending=False)

                col_map = {
                    "symbol": "Symbol",
                    "description": "Description",
                    date_col: "Settle date",
                    cur_col if cur_col else "": "Currency",
                    "amount": "Amount",
                }
                d_disp.rename(columns=col_map, inplace=True)

                if "Amount" in d_disp.columns:
                    d_disp["Amount"] = pd.to_numeric(d_disp["Amount"], errors="coerce").round(2)

                st.dataframe(
                    d_disp,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Symbol":      st.column_config.TextColumn(),
                        "Description": st.column_config.TextColumn(),
                        "Settle date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                        "Currency":    st.column_config.TextColumn(),
                        "Amount":      st.column_config.NumberColumn(format="%.2f"),
                    },
                )

# ========================= PAGE: Transactions =========================
elif page == "📈 Transactions":
    st.header("Transactions overview")

    if df_tx.empty:
        st.warning("No transactions in the table.")
    else:
        st.dataframe(df_tx)

# ========================= PAGE: Open option positions =========================
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

            # Keep only options
            df_opt["assetclass"] = df_opt["assetclass"].astype(str).str.upper().str.strip()
            df_opt = df_opt[df_opt["assetclass"] == "OPT"]

            # Normalize numeric columns
            for col in ["quantity", "tradeprice", "strike"]:
                if col in df_opt.columns:
                    df_opt[col] = pd.to_numeric(df_opt[col], errors="coerce").fillna(0.0)

            # Normalize side/type
            if "buy/sell" in df_opt.columns:
                df_opt["buy/sell"] = df_opt["buy/sell"].astype(str).str.upper().str.strip()
            if "put/call" in df_opt.columns:
                df_opt["put/call"] = df_opt["put/call"].astype(str).str.upper().str.strip()

            # Currency column
            CUR_COL = "currencyprimary" if "currencyprimary" in df_opt.columns else ("currency" if "currency" in df_opt.columns else None)
            if CUR_COL is not None:
                df_opt[CUR_COL] = df_opt[CUR_COL].astype(str).str.upper().str.strip()

            # Net filter by description
            net_by_desc = (
                df_opt.groupby("description", as_index=False)["quantity"]
                .sum()
                .rename(columns={"quantity": "net_quantity"})
            )
            df_opt = df_opt.merge(net_by_desc, on="description", how="left")
            df_opt = df_opt[df_opt["net_quantity"].round(8) != 0]

            # Premiums
            df_opt["premium"] = (df_opt["tradeprice"] * df_opt["quantity"] * 100).round(2)
            df_opt["unearned_premium"] = (-df_opt["premium"]).round(2)

            # Expiry -> DTE
            def to_date_yyyymmdd(x):
                if pd.isna(x):
                    return pd.NaT
                s = str(x).strip().split(".")[0]
                s = s[:8]
                return pd.to_datetime(s, format="%Y%m%d", errors="coerce")

            if "expiry" in df_opt.columns:
                df_opt["expiry_dt"] = df_opt["expiry"].apply(to_date_yyyymmdd)
                today = pd.Timestamp.today().normalize()
                df_opt["DTE"] = (df_opt["expiry_dt"] - today).dt.days
                df_opt["DTE"] = df_opt["DTE"].fillna(0).clip(lower=0).astype(int)
            else:
                df_opt["expiry_dt"] = pd.NaT
                df_opt["DTE"] = None

            # Capital blocked (signed)
            df_opt["capital blocked"] = (df_opt["strike"] * df_opt["quantity"] * 100).round(2)

            # Put/Call labels
            if "put/call" in df_opt.columns:
                df_opt["put/call"] = df_opt["put/call"].map({"C": "Call", "P": "Put"}).fillna(df_opt["put/call"])

            # --------- Two-column layout
            left, right = st.columns([2.6, 1.4])

            # ===================== LEFT SIDE =====================
            with left:
                display_cols = [
                    "description","put/call","quantity","strike","premium","unearned_premium",
                    "DTE","capital blocked"
                ]
                if CUR_COL is not None:
                    display_cols.append(CUR_COL)
                display_cols = [c for c in display_cols if c in df_opt.columns]

                sort_cols = [c for c in ["DTE", "description"] if c in df_opt.columns]
                if sort_cols:
                    df_opt = df_opt.sort_values(sort_cols, ascending=[True, True] if len(sort_cols) == 2 else True)

                st.subheader("Positions")
                col_config = {
                    "description":        st.column_config.TextColumn("description"),
                    "put/call":           st.column_config.TextColumn("put/call"),
                    "quantity":           st.column_config.NumberColumn("quantity", format="%.2f"),
                    "strike":             st.column_config.NumberColumn("strike", format="%.2f"),
                    "premium":            st.column_config.NumberColumn("premium", format="%.2f"),
                    "unearned_premium":   st.column_config.NumberColumn("unearned premium", format="%.2f"),
                    "DTE":                st.column_config.NumberColumn("DTE", format="%d"),
                    "capital blocked":    st.column_config.NumberColumn("capital blocked", format="%.2f"),
                }
                if CUR_COL is not None:
                    col_config[CUR_COL] = st.column_config.TextColumn("currency")

                st.dataframe(
                    df_opt[display_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config=col_config,
                )

            # ===================== RIGHT SIDE =====================
            with right:
                st.subheader("Summary")

                def _count(side, opttype):
                    mask = (df_opt.get("buy/sell") == side) & (df_opt.get("put/call") == opttype)
                    return int(mask.sum())

                counts_df = pd.DataFrame({
                    "Metric": [
                        "Count of positions (sell put)",
                        "Count of positions (sell call)",
                        "Count of positions (buy put)",
                        "Count of positions (buy call)",
                    ],
                    "Value": [
                        _count("SELL", "Put"),
                        _count("SELL", "Call"),
                        _count("BUY",  "Put"),
                        _count("BUY",  "Call"),
                    ],
                })
                st.dataframe(
                    counts_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={"Metric": st.column_config.TextColumn(),
                                   "Value":  st.column_config.NumberColumn(format="%d")},
                    height=190,
                )

            # (omitted visuals and by-expiry tables for brevity — keep your previous blocks if needed)

            st.markdown("---")
            show_details = st.checkbox("Show details")
            if show_details and not df_opt.empty:
                sel = st.selectbox("description:", options=sorted(df_opt["description"].unique().tolist()))
                details = df_tx.copy()
                details.columns = [c.lower() for c in details.columns]
                details = details[details.get("description") == sel]

                date_candidates = [c for c in ("date", "tradedate", "settledate", "lasttradingday") if c in details.columns]
                if date_candidates:
                    sort_col = date_candidates[0]
                    with pd.option_context('mode.chained_assignment', None):
                        try:
                            details[sort_col] = pd.to_datetime(details[sort_col], errors="coerce")
                        except Exception:
                            pass
                    details = details.sort_values(sort_col, ascending=False)

                st.dataframe(details, use_container_width=True, hide_index=True)

# ========================= PAGE: Open stock positions =========================
elif page == "Open stock positions":
    st.header("Open stock positions")

    if df_tx.empty:
        st.warning("No transactions in the table.")
    else:
        # 1) Normalize and validate
        df = df_tx.copy()
        df.columns = [c.lower() for c in df.columns]

        sym_col = "symbol" if "symbol" in df.columns else ("underlyingsymbol" if "underlyingsymbol" in df.columns else None)
        cur_col = "currencyprimary" if "currencyprimary" in df.columns else ("currency" if "currency" in df.columns else None)

        required = ["assetclass", "quantity", "tradeprice"]
        missing = [c for c in required if c not in df.columns]
        if sym_col is None:
            missing.append("symbol (or underlyingsymbol)")
        if missing:
            st.error(f"Missing columns in transactions table: {', '.join(missing)}")
            st.stop()

        # 2) Keep only STK
        df["assetclass"] = df["assetclass"].astype(str).str.upper().str.strip()
        df = df[df["assetclass"] == "STK"].copy()
        if df.empty:
            st.info("No stock transactions found.")
            st.stop()

        # 3) Normalize fields
        df["quantity"]   = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
        df["tradeprice"] = pd.to_numeric(df["tradeprice"], errors="coerce").fillna(0.0)
        if cur_col:
            df[cur_col] = df[cur_col].astype(str).str.upper().str.strip()
        if "description" not in df.columns:
            df["description"] = ""

        # 4) SIGNED quantity (BUY = +, SELL = −). If buy/sell missing, assume already signed.
        if "buy/sell" in df.columns:
            side = df["buy/sell"].astype(str).str.upper().str.strip()
            df["signed_qty"] = np.where(side == "SELL", -df["quantity"].abs(), df["quantity"].abs())
        else:
            df["signed_qty"] = df["quantity"]

        # 5) Symbols with non-zero net quantity
        net_qty = (
            df.groupby(sym_col, dropna=False)["signed_qty"]
              .sum()
              .rename("net_qty")
              .reset_index()
        )
        open_syms = net_qty[net_qty["net_qty"].round(8) != 0][sym_col].tolist()
        if not open_syms:
            st.success("All stock positions are closed (net quantity = 0 for every symbol).")
            st.stop()

        # 6) Keep rows for open symbols
        df_open = df[df[sym_col].isin(open_syms)].copy()
        df_open["signed_amt"] = df_open["tradeprice"] * df_open["signed_qty"]

        # Representative description per symbol
        desc_map = (
            df_open[[sym_col, "description"]]
            .dropna(subset=["description"])
            .groupby(sym_col, as_index=False)
            .agg({"description": "first"})
        )

        # 7) Aggregate by symbol (+currency)
        group_cols = [sym_col] + ([cur_col] if cur_col else [])
        agg = (
            df_open.groupby(group_cols, dropna=False)
                   .agg(
                       Qty=("signed_qty", "sum"),
                       Cost_base=("signed_amt", "sum"),
                   )
                   .reset_index()
        )

        # 8) Avg price
        agg["Avg_price"] = np.where(agg["Qty"].round(8) != 0, agg["Cost_base"] / agg["Qty"], np.nan)

        # attach description
        agg = agg.merge(desc_map, on=sym_col, how="left")

        # rename for display
        agg.rename(columns={
            sym_col: "Symbol",
            "description": "Description",
            cur_col if cur_col else "": "currencyprimary",
        }, inplace=True)

        # 9) Fetch Yahoo EOD current prices and compute P&L
        prices_df = fetch_eod_close(agg["Symbol"].tolist())
        agg = agg.merge(prices_df, on="Symbol", how="left")

        # Unrealized P&L = (Current price - Avg_price) * Qty
        agg["unrealized pnl"] = (pd.to_numeric(agg["Current price"], errors="coerce") - agg["Avg_price"]) * agg["Qty"]

        # --- Unrealized PnL % = unrealized pnl / cost base (x100), safe divide
        agg["unrealized pnl %"] = np.where(
            agg["Cost_base"].abs() > 0,
            (agg["unrealized pnl"] / agg["Cost_base"]) * 100.0,
            np.nan
        )

        # Order columns as desired
        display_cols = [
            "Symbol", "Description", "currencyprimary", "Cost_base", "Qty",
            "Avg_price", "Current price", "unrealized pnl", "unrealized pnl %"
        ]
        display_cols = [c for c in display_cols if c in agg.columns]
        agg = agg[display_cols].sort_values("Symbol").reset_index(drop=True)

        # === LAST STEP: round numbers to 2 decimals only for display ===
        agg_disp = agg.copy()
        num_cols = agg_disp.select_dtypes(include=["number"]).columns
        agg_disp[num_cols] = agg_disp[num_cols].round(2)

        st.dataframe(
            agg_disp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol":            st.column_config.TextColumn(),
                "Description":       st.column_config.TextColumn(),
                "currencyprimary":   st.column_config.TextColumn("currencyprimary"),
                "Cost_base":         st.column_config.NumberColumn(format="%.2f"),
                "Qty":               st.column_config.NumberColumn(format="%.2f"),
                "Avg_price":         st.column_config.NumberColumn(format="%.2f"),
                "Current price":     st.column_config.NumberColumn(format="%.2f"),
                "unrealized pnl":    st.column_config.NumberColumn(format="%.2f"),
                "unrealized pnl %":  st.column_config.NumberColumn(format="%.2f%%"),
            }
        )

        # Helper expander: net-qty pivot
        with st.expander("Show net quantity per symbol (pivot-style)"):
            st.dataframe(
                net_qty.rename(columns={sym_col: "Symbol"}),
                use_container_width=True,
                hide_index=True
            )

# ========================= PAGE: Settings =========================
else:
    st.header("Settings")
    st.info("Configuration options will be here.")
