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
VIEW_REALIZED_FIFO = st.secrets["VIEW_REALIZED_FIFO"]
VIEW_REALIZED_FIFO_USD = st.secrets["VIEW_REALIZED_FIFO_USD"]

# --- refactor
@st.cache_resource
def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)

engine = get_engine()

# --- SIDEBAR
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ("📊 Dividends Overview",
     "📈 Transactions",
     "Open option positions",
     "Open stock positions",
     "📒 Closed positions / realized PnL (FIFO, USD)",
     "📊 Realized PnL Analysis",
     "Option ROI Calculator",
     "⚙️ Settings",
    ),
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
    df = pd.read_sql(f"SELECT * FROM {TABLE_DIVI}", engine)
    df.columns = [c.lower() for c in df.columns]
    return df

@st.cache_data(ttl=600)
def load_transactions() -> pd.DataFrame:
    df = pd.read_sql(f"SELECT * FROM {TABLE_TRANSACTIONS}", engine)
    df.columns = [c.lower() for c in df.columns]
    return df

df_divi = load_dividends()
df_tx   = load_transactions()

@st.cache_data(ttl=3600)
def fetch_eod_close(symbols: list[str]) -> pd.DataFrame:
    """
    Fetch latest available EOD Close for each Yahoo ticker in `symbols`.
    Returns DataFrame: columns = ['Symbol', 'Current price'].
    """
    if yf is None:
        return pd.DataFrame({"Symbol": symbols, "Current price": [np.nan]*len(symbols)})

    # unique, non-empty
    tickers = [s for s in dict.fromkeys([str(s).strip() for s in symbols]) if s]
    if not tickers:
        return pd.DataFrame(columns=["Symbol", "Current price"])

    # Pull a short recent window to ensure we get a non-NaN close (e.g., weekends/holidays)
    data = yf.download(
        tickers=" ".join(tickers),
        period="10d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    # tickery, ktoré chceme deliť 100 (GBp -> základná mena)
    div_by_100 = { "BYG.L" }
    div_by_100_upper = {s.upper() for s in div_by_100}

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

        # --- konverzia hneď po stiahnutí ---
        if t.upper() in div_by_100_upper and pd.notna(price):
            price = price / 100.0

        closes.append({"Symbol": t, "Current price": price})

    return pd.DataFrame(closes)

@st.cache_data(ttl=300)
def load_realized(view_name: str) -> pd.DataFrame:
    df = pd.read_sql(f"SELECT * FROM {view_name}", engine)
    df.columns = [c.lower() for c in df.columns]
    return df

# --- Basic cleanup for dividends
if not df_divi.empty:
    df_divi["amount"] = pd.to_numeric(df_divi.get("amount", 0), errors="coerce")
    df_divi.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_divi["amount"] = df_divi["amount"].fillna(0)

# --- PAGE: Dividends Overview
if page == "📊 Dividends Overview":
    st.title("Dividends overview")

    # Always define columns (fixes Pylance: col2 not defined)
    col1, col2 = st.columns([2.7, 1.3])

    if df_divi.empty:
        with col1:
            st.warning("The dividends table is empty.")
        with col2:
            st.info("No data to display.")
    else:
        # Dates & helpers
        df_divi["settledate"] = pd.to_datetime(df_divi["settledate"], format="%Y%m%d", errors="coerce")
        df_divi["month"] = df_divi["settledate"].dt.strftime("%b")
        df_divi["year"] = df_divi["settledate"].dt.year
        df_divi["settledate_str"] = df_divi["settledate"].dt.strftime("%m/%d/%Y")

        current_period = pd.Timestamp.today().to_period("M")
        df_divi_cur = df_divi[df_divi["settledate"].dt.to_period("M") == current_period]

        df_show = (
            df_divi_cur.sort_values("settledate", ascending=False)[["symbol", "settledate_str", "currency", "amount"]]
            .reset_index(drop=True)
        )

        # ----------------------- LEFT COLUMN -----------------------
        with col1:
            tab1, tab2, tab3 = st.tabs(["📅 Year", "🗓️ By Quarter", "🔖 Ticker"])

            # ----- Tab 1: Year
            with tab1:
                st.subheader("Summary by Year & Currency")
                summary = df_divi.groupby(["year", "currency"], dropna=False)["amount"].sum().reset_index()

                chart = (
                    alt.Chart(summary)
                    .mark_bar()
                    .encode(
                        x=alt.X("year:O", title="Year"),
                        y=alt.Y("amount:Q", title="Sum of Dividends"),
                        color=alt.Color("currency:N", title="Currency"),
                        tooltip=["year", "currency", "amount"],
                    )
                    .properties(width=CHART_WIDTH_LEFT)
                )
                st.altair_chart(chart, use_container_width=False)

                st.subheader("Year × Currency")
                summary_y = df_divi.groupby(["year", "currency"], as_index=False)["amount"].sum()
                pivot_y = summary_y.pivot_table(index="currency", columns="year", values="amount", aggfunc="sum", fill_value=0)

                year_cols = sorted(pivot_y.columns.tolist())
                pivot_y = pivot_y.reindex(columns=year_cols)

                row_total = pivot_y[year_cols].sum().to_frame().T
                row_total.index = ["Total"]
                pivot_y = pd.concat([pivot_y, row_total], axis=0)

                pivot_y.index.name = "Currency"
                pivot_y.columns.name = "Year"
                display_df = pivot_y.reset_index()

                num_cols = [c for c in display_df.columns if c != "Currency"]
                display_df[num_cols] = (
                    display_df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).round(0).astype(int)
                )

                st.dataframe(
                    display_df,
                    width=CHART_WIDTH_LEFT,
                    use_container_width=False,
                    hide_index=True,
                    height=min(420, 42 * (len(display_df) + 1)),
                    column_config={
                        "Currency": st.column_config.TextColumn(),
                        **{c: st.column_config.NumberColumn(format="%,d") for c in num_cols},
                    },
                )

            # ----- Tab 2: Quarter
            with tab2:
                st.subheader("Summary by Quarter & Currency")
                year_options = sorted(df_divi["year"].dropna().unique())
                selected_year = st.selectbox(
                    "Select year:",
                    options=year_options,
                    index=len(year_options) - 1 if len(year_options) else 0,
                    key="sel_year",
                )
                st.markdown(
                    f"""
                    <style>
                    div[data-baseweb="select"] > div {{ width: {CHART_WIDTH_TAB2}px !important; }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                df_y = df_divi[df_divi["year"] == selected_year].copy()
                df_y["quarter"] = "Q" + df_y["settledate"].dt.quarter.astype(str)

                summary_q = df_y.groupby(["quarter", "currency"], as_index=False, dropna=False)["amount"].sum()
                order_q = ["Q1", "Q2", "Q3", "Q4"]
                summary_q["quarter"] = pd.Categorical(summary_q["quarter"], categories=order_q, ordered=True)
                summary_q = summary_q.sort_values("quarter")

                chart1 = (
                    alt.Chart(summary_q)
                    .mark_bar()
                    .encode(
                        x=alt.X("quarter:O", sort=order_q, title="Quarter"),
                        y=alt.Y("amount:Q", title="Sum"),
                        color=alt.Color("currency:N", title="Currency"),
                        tooltip=["quarter", "currency", "amount"],
                    )
                    .properties(width=CHART_WIDTH_TAB2)
                )
                st.altair_chart(chart1, use_container_width=False)

                pivot = (
                    summary_q.assign(quarter=pd.Categorical(summary_q["quarter"], categories=order_q, ordered=True))
                    .pivot_table(index="currency", columns="quarter", values="amount", aggfunc="sum", fill_value=0)
                    .reindex(columns=order_q)
                )

                pivot["Total"] = pivot.sum(axis=1)
                total_row = pivot.sum(axis=0).to_frame().T
                total_row.index = ["Total"]
                pivot = pd.concat([pivot, total_row], axis=0)

                pivot.index.name = "Currency"
                pivot.columns.name = "Quarter"

                st.subheader(f"Quarter × Currency – {selected_year}")
                display_df = pivot.reset_index()

                st.dataframe(
                    display_df,
                    width=CHART_WIDTH_TAB2,
                    use_container_width=False,
                    hide_index=True,
                    height=min(380, 42 * (len(display_df) + 1)),
                    column_config={**{c: st.column_config.NumberColumn(format="%.2f") for c in order_q + ["Total"]}},
                )

            # ----- Tab 3: Ticker
            with tab3:
                st.subheader("Summary by Ticker & Year")
                tics = sorted(df_divi["symbol"].dropna().unique())
                sel_t = st.multiselect("Choose ticker(s):", options=tics, default=tics[:1], key="sel_t")

                if sel_t:
                    df_t = df_divi[df_divi["symbol"].isin(sel_t)]
                    summ_t = df_t.groupby(["year", "symbol"])["amount"].sum().reset_index()

                    chart2 = (
                        alt.Chart(summ_t)
                        .mark_bar()
                        .encode(
                            x=alt.X("year:O", title="Year"),
                            y=alt.Y("amount:Q"),
                            color=alt.Color("symbol:N", title="Ticker"),
                            tooltip=["year", "symbol", "amount"],
                        )
                        .properties(width=CHART_WIDTH_TAB3)
                    )
                    st.altair_chart(chart2, use_container_width=False)

                    year_totals = df_t.groupby("year", as_index=False)["amount"].sum()
                    tmp = year_totals.sort_values("year")
                    tmp["change"] = tmp["amount"].diff().fillna(0)
                    year_totals = tmp.sort_values("year", ascending=False)

                    st.dataframe(
                        year_totals.rename(columns={"year": "Year", "amount": "Total", "change": "Change"}),
                        width=CHART_WIDTH_TAB3,
                        use_container_width=False,
                        height=min(320, 42 * (len(year_totals) + 1)),
                        column_config={
                            "Year": st.column_config.NumberColumn(format="%d"),
                            "Total": st.column_config.NumberColumn(format="%.2f"),
                            "Change": st.column_config.NumberColumn(format="%.2f"),
                        },
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
                    "Total": st.column_config.NumberColumn(format="%.0f"),
                },
            )
# ========================= PAGE: Transactions =========================
elif page == "📈 Transactions":
    st.header("Transactions overview")

    if df_tx.empty: 
        st.warning("No transactions in the table.") 
    else: 
        st.dataframe(df_tx)
# ========================= PAGE: Closed positions / Realized PnL USD =========================
elif page == "📒 Closed positions / realized PnL (FIFO, USD)":
    st.header("Closed positions / realized PnL (FIFO, USD)")

    df_rlz = load_realized(VIEW_REALIZED_FIFO_USD)

    if df_rlz.empty:
        st.info("No realized lot matches in this view yet.")
        st.stop()

    # --- normalize dates
    for c in ["open_date", "close_date", "created_at"]:
        if c in df_rlz.columns:
            df_rlz[c] = pd.to_datetime(df_rlz[c], errors="coerce")

    # --- filters
    c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.0, 1.0])

    with c1:
        tickers = sorted(df_rlz["ticker"].dropna().unique().tolist()) if "ticker" in df_rlz.columns else []
        sel_ticker = st.multiselect("Ticker", options=tickers, default=[])

    with c2:
        asset_opts = sorted(df_rlz["asset_class"].dropna().unique().tolist()) if "asset_class" in df_rlz.columns else []
        sel_asset = st.multiselect("Asset class", options=asset_opts, default=[])

    with c3:
        date_rng = None
        if "close_date" in df_rlz.columns and df_rlz["close_date"].notna().any():
            min_d = df_rlz["close_date"].min().date()
            max_d = df_rlz["close_date"].max().date()
            date_rng = st.date_input("Close date range", value=(min_d, max_d))
        else:
            st.caption("No close_date values available.")

    with c4:
        group_mode = st.selectbox("Group by", ["None", "Month", "Year"], index=1)

    df_f = df_rlz.copy()
    if sel_ticker and "ticker" in df_f.columns:
        df_f = df_f[df_f["ticker"].isin(sel_ticker)]
    if sel_asset and "asset_class" in df_f.columns:
        df_f = df_f[df_f["asset_class"].isin(sel_asset)]
    if date_rng and "close_date" in df_f.columns and len(date_rng) == 2 and date_rng[0] and date_rng[1]:
        start = pd.to_datetime(date_rng[0])
        end = pd.to_datetime(date_rng[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df_f = df_f[(df_f["close_date"] >= start) & (df_f["close_date"] <= end)]

    # --- numeric cleanup
    for col in [
        "realized_local", "realized_usd", "commission_local", "commission_usd",
        "qty_matched", "open_price", "close_price"
    ]:
        if col in df_f.columns:
            df_f[col] = pd.to_numeric(df_f[col], errors="coerce")

    # --- KPI
    realized_local_total = pd.to_numeric(df_f.get("realized_local", 0), errors="coerce").fillna(0).sum()
    realized_usd_total = pd.to_numeric(df_f.get("realized_usd", 0), errors="coerce").fillna(0).sum()
    commission_local_total = pd.to_numeric(df_f.get("commission_local", 0), errors="coerce").fillna(0).sum()
    commission_usd_total = pd.to_numeric(df_f.get("commission_usd", 0), errors="coerce").fillna(0).sum()

    trades_cnt = len(df_f)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Realized g/l local ccy", f"{realized_local_total:,.2f}")
    k2.metric("Realized g/l USD", f"{realized_usd_total:,.2f}")
    k3.metric("Commission local", f"{commission_local_total:,.2f}")
    k4.metric("Commission USD", f"{commission_usd_total:,.2f}")

    st.caption(f"Rows: {trades_cnt}")

    # --- graph removed for now

    st.subheader("Details")

    display_cols = [
        "instrument",
        "qty_matched",
        "open_date",
        "close_date",
        "open_price",
        "close_price",
        "currency_local",
        "realized_local",
        "realized_usd",
        "commission_local",
        "commission_usd",
    ]
    display_cols = [c for c in display_cols if c in df_f.columns]

    df_disp = df_f[display_cols].copy()

    for c in [
        "qty_matched", "open_price", "close_price",
        "realized_local", "realized_usd",
        "commission_local", "commission_usd"
    ]:
        if c in df_disp.columns:
            df_disp[c] = pd.to_numeric(df_disp[c], errors="coerce").round(4)

    sort_cols = [c for c in ["close_date", "instrument"] if c in df_disp.columns]
    if sort_cols:
        ascending = [False, True] if len(sort_cols) == 2 else [False]
        df_disp = df_disp.sort_values(sort_cols, ascending=ascending)

    st.dataframe(
        df_disp,
        use_container_width=True,
        hide_index=True
    )

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

            # --- Normalize currency column & detect its name
            # We support either `currencyprimary` or `currency`
            CUR_COL = "currencyprimary" if "currencyprimary" in df_opt.columns else ("currency" if "currency" in df_opt.columns else None)
            if CUR_COL is not None:
                df_opt[CUR_COL] = df_opt[CUR_COL].astype(str).str.upper().str.strip()

            # Net filter: keep descriptions with non-zero net quantity
            net_by_desc = (
                df_opt.groupby("description", as_index=False)["quantity"]
                .sum()
                .rename(columns={"quantity": "net_quantity"})
            )
            df_opt = df_opt.merge(net_by_desc, on="description", how="left")
            df_opt = df_opt[df_opt["net_quantity"].round(8) != 0]

            #
        
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

            df_opt = df_opt[df_opt["quantity"].round(8) != 0].copy()

            # Put/Call labels
            if "put/call" in df_opt.columns:
                df_opt["put/call"] = df_opt["put/call"].map({"C": "Call", "P": "Put"}).fillna(df_opt["put/call"])

            # ------------------------------------------------------------
            # AGGREGATE OPEN OPTION POSITIONS
            # merge multiple fills of the same contract into 1 row
            # ------------------------------------------------------------
            group_cols = ["description", "buy/sell", "put/call", "strike", "expiry_dt"]
            if CUR_COL is not None:
                group_cols.append(CUR_COL)

            df_opt = (
                df_opt.groupby(group_cols, dropna=False, as_index=False)
                    .agg(
                        quantity=("quantity", "sum"),
                        tradeprice=("tradeprice", "mean"),          # optional: len pre info
                        premium=("premium", "sum"),
                        unearned_premium=("unearned_premium", "sum"),
                        DTE=("DTE", "min"),
                    )
            )

            # keep only truly open positions
            df_opt = df_opt[df_opt["quantity"].round(8) != 0].copy()

            # recompute capital blocked on contract level (clean & consistent)
            df_opt["capital blocked"] = (df_opt["strike"] * df_opt["quantity"] * 100).round(2)

            # nice ordering
            sort_cols = [c for c in ["DTE", "description", "strike"] if c in df_opt.columns]
            if sort_cols:
                df_opt = df_opt.sort_values(sort_cols, ascending=True)

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

                # Sort by expiry (DTE) then by description
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

                # ======== VISUALS ========
                st.markdown("### Visuals")

                # --- Currency filter (affects all visuals below)
                cur_options = sorted(df_opt.get(CUR_COL, pd.Series(dtype=str)).dropna().unique().tolist()) if CUR_COL else []
                selected_curs = st.multiselect(
                    "Filter by currency (affects pies and charts below):",
                    options=cur_options,
                    default=cur_options if cur_options else None,
                    key="cur_filter_openopts"
                )

                if selected_curs and CUR_COL:
                    df_vis = df_opt[df_opt[CUR_COL].isin(selected_curs)].copy()
                else:
                    df_vis = df_opt.copy()

                # Short puts subset (for capital blocked)
                df_sp_vis = df_vis[(df_vis.get("buy/sell") == "SELL") & (df_vis.get("put/call") == "Put")].copy()

                # --- Aggregations for pies
                if CUR_COL:
                    up_by_cc = (
                        df_vis.groupby(CUR_COL, dropna=False)["unearned_premium"]
                              .sum().reset_index().rename(columns={CUR_COL: "currency", "unearned_premium": "up_sum"})
                    )
                    cb_by_cc = (
                        df_sp_vis.groupby(CUR_COL, dropna=False)["capital blocked"]
                                 .apply(lambda s: s.abs().sum()).reset_index()
                                 .rename(columns={CUR_COL: "currency", "capital blocked": "cb_sum"})
                    )
                else:
                    # Fallback if no currency column exists
                    up_by_cc = pd.DataFrame({"currency": ["N/A"], "up_sum": [df_vis["unearned_premium"].sum()]})
                    cb_by_cc = pd.DataFrame({"currency": ["N/A"], "cb_sum": [df_sp_vis["capital blocked"].abs().sum()]})

                # --- Percent shares by EXPIRY: share of total unearned premium (within selected currencies)
                exp_agg = (
                    df_vis.groupby("expiry_dt", dropna=False)["unearned_premium"]
                          .sum().rename("up_sum").reset_index()
                )
                exp_agg = exp_agg.sort_values("expiry_dt")
                exp_agg["expiry_str"] = exp_agg["expiry_dt"].dt.strftime("%Y-%m-%d").fillna("N/A")
                total_up_exp = exp_agg["up_sum"].sum()
                exp_agg["share_pct"] = np.where(total_up_exp > 0, exp_agg["up_sum"] / total_up_exp * 100, np.nan)

                # --- Percent shares by TICKER: share of total unearned premium (within selected currencies)
                if "underlyingsymbol" in df_vis.columns and df_vis["underlyingsymbol"].notna().any():
                    tick_col = "underlyingsymbol"
                else:
                    tick_col = "ticker_fallback"
                    df_vis[tick_col] = df_vis["description"].astype(str).str.split().str[0]

                tic_agg = (
                    df_vis.groupby(tick_col, dropna=False)["unearned_premium"]
                          .sum().rename("up_sum").reset_index()
                          .rename(columns={tick_col: "ticker"})
                          .sort_values("ticker")
                )
                total_up_tic = tic_agg["up_sum"].sum()
                tic_agg["share_pct"] = np.where(total_up_tic > 0, tic_agg["up_sum"] / total_up_tic * 100, np.nan)

                # --- Row 1: smaller pies (blue & light purple)
                c1, c2 = st.columns(2)

                with c1:
                    st.subheader("Unearned premium by currency")
                    pie1 = (
                        alt.Chart(up_by_cc)
                        .mark_arc(innerRadius=60)
                        .encode(
                            theta=alt.Theta("up_sum:Q", title="Unearned premium"),
                            color=alt.Color("currency:N", scale=alt.Scale(scheme="blues"), legend=None),
                            tooltip=[alt.Tooltip("currency:N"), alt.Tooltip("up_sum:Q", format=",.2f")]
                        )
                        .properties(width=260, height=220)
                    )
                    st.altair_chart(pie1, use_container_width=False)

                with c2:
                    st.subheader("Capital blocked by currency (short puts)")
                    pie2 = (
                        alt.Chart(cb_by_cc)
                        .mark_arc(innerRadius=60)
                        .encode(
                            theta=alt.Theta("cb_sum:Q", title="Capital blocked"),
                            color=alt.Color("currency:N", scale=alt.Scale(scheme="purples"), legend=None),
                            tooltip=[alt.Tooltip("currency:N"), alt.Tooltip("cb_sum:Q", format=",.2f")]
                        )
                        .properties(width=260, height=220)
                    )
                    st.altair_chart(pie2, use_container_width=False)

                # --- Row 2: bars (left axis = sum, right axis = % share)
                l1, l2 = st.columns(2)

                with l1:
                    st.subheader("By expiry — sum & share % (currency filter applied)")
                    base = alt.Chart(exp_agg).encode(x=alt.X("expiry_str:N", title="Expiry")).properties(height=320)
                    bars = base.mark_bar().encode(
                        y=alt.Y("up_sum:Q", title="Unearned premium (sum)", axis=alt.Axis(format=",.0f")),
                        tooltip=[
                            alt.Tooltip("expiry_str:N", title="Expiry"),
                            alt.Tooltip("up_sum:Q", title="Sum", format=",.2f"),
                            alt.Tooltip("share_pct:Q", title="Share %", format=",.2f"),
                        ],
                    )
                    line = base.mark_line(point=True, color="red").encode(
                        y=alt.Y("share_pct:Q", title="Share of total (%)")
                    )
                    st.altair_chart(alt.layer(bars, line).resolve_scale(y='independent'), use_container_width=True)

                with l2:
                    st.subheader("By ticker — sum & share % (currency filter applied)")
                    base_t = alt.Chart(tic_agg).encode(x=alt.X("ticker:N", title="Ticker")).properties(height=320)
                    bars_t = base_t.mark_bar().encode(
                        y=alt.Y("up_sum:Q", title="Unearned premium (sum)", axis=alt.Axis(format=",.0f")),
                        tooltip=[
                            alt.Tooltip("ticker:N", title="Ticker"),
                            alt.Tooltip("up_sum:Q", title="Sum", format=",.2f"),
                            alt.Tooltip("share_pct:Q", title="Share %", format=",.2f"),
                        ],
                    )
                    line_t = base_t.mark_line(point=True, color="red").encode(
                        y=alt.Y("share_pct:Q", title="Share of total (%)")
                    )
                    st.altair_chart(alt.layer(bars_t, line_t).resolve_scale(y='independent'), use_container_width=True)

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

                # Capital blocked & unearned premium totals by currency (show USD/EUR if present)
                if CUR_COL:
                    df_sp_all = df_opt[(df_opt.get("buy/sell") == "SELL") & (df_opt.get("put/call") == "Put")].copy()
                    cap_tot = (
                        df_sp_all.groupby(CUR_COL, dropna=False)["capital blocked"]
                                 .apply(lambda s: s.abs().sum())
                    )
                    up_tot = (
                        df_opt.groupby(CUR_COL, dropna=False)["unearned_premium"]
                              .sum()
                    )
                    # Try to present in common order
                    order = [c for c in ["USD", "EUR"] if c in cap_tot.index.union(up_tot.index)]
                    cap_tot = cap_tot.reindex(order).fillna(0.0).round(2)
                    up_tot  = up_tot.reindex(order).fillna(0.0).round(2)

                    totals_df = pd.DataFrame({
                        "Metric": [f"Capital blocked total ({c})" for c in order] +
                                  [f"Unearned premium total ({c})" for c in order],
                        "Value":  list(cap_tot.values) + list(up_tot.values),
                    })
                else:
                    totals_df = pd.DataFrame({
                        "Metric": ["Capital blocked total", "Unearned premium total"],
                        "Value":  [df_opt["capital blocked"].abs().sum(), df_opt["unearned_premium"].sum()],
                    })

                st.dataframe(
                    totals_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={"Metric": st.column_config.TextColumn(),
                                   "Value":  st.column_config.NumberColumn(format="%.2f")},
                    height=220,
                )

                # Tables by expiry (kept)
                exp_sp = df_opt.copy()
                exp_sp["expiry_str"] = exp_sp["expiry_dt"].dt.strftime("%Y-%m-%d")
                if CUR_COL:
                    cap_by_exp = (
                        exp_sp[exp_sp.get("buy/sell").eq("SELL") & exp_sp.get("put/call").eq("Put")]
                        .groupby(["expiry_str", CUR_COL], dropna=False)["capital blocked"]
                        .apply(lambda s: s.abs().sum())
                        .unstack(CUR_COL)
                        .fillna(0.0).round(2)
                        .reset_index()
                        .rename(columns={"expiry_str": "expiry"})
                    )
                    up_by_exp = (
                        exp_sp.groupby(["expiry_str", CUR_COL], dropna=False)["unearned_premium"]
                        .sum()
                        .unstack(CUR_COL)
                        .fillna(0.0).round(2)
                        .reset_index()
                        .rename(columns={"expiry_str": "expiry"})
                    )
                else:
                    cap_by_exp = (
                        exp_sp[exp_sp.get("buy/sell").eq("SELL") & exp_sp.get("put/call").eq("Put")]
                        .groupby(["expiry_str"], dropna=False)["capital blocked"]
                        .apply(lambda s: s.abs().sum())
                        .reset_index().rename(columns={"capital blocked":"Total"})
                    )
                    up_by_exp = (
                        exp_sp.groupby(["expiry_str"], dropna=False)["unearned_premium"]
                        .sum().reset_index().rename(columns={"unearned_premium":"Total"})
                    )

                st.markdown("**Capital blocked by expiry**")
                st.dataframe(cap_by_exp, hide_index=True, use_container_width=True)

                st.markdown("**Unearned premium by expiry**")
                st.dataframe(up_by_exp, hide_index=True, use_container_width=True)

            # --------- SHOW DETAILS (by description)
            st.markdown("---")
            show_details = st.checkbox("Show details")
            if show_details and not df_opt.empty:
                sel = st.selectbox("description:", options=sorted(df_opt["description"].unique().tolist()))
                details = df_tx.copy()
                details.columns = [c.lower() for c in details.columns]
                details = details[details.get("description") == sel]

                # Sort by best available date column
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

        # 4) Build SIGNED quantity (BUY = +, SELL = −). If buy/sell is missing, assume quantity already signed.
        if "buy/sell" in df.columns:
            side = df["buy/sell"].astype(str).str.upper().str.strip()
            df["signed_qty"] = np.where(side == "SELL", -df["quantity"].abs(), df["quantity"].abs())
        else:
            df["signed_qty"] = df["quantity"]

        # 5) First find symbols with non-zero net quantity (pivot logic)
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

        # 6) Work only with rows for those symbols
        df_open = df[df[sym_col].isin(open_syms)].copy()
        df_open["signed_amt"] = df_open["tradeprice"] * df_open["signed_qty"]

        # Representative description per symbol (first non-null)
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

        # Order columns as desired
        display_cols = ["Symbol", "Description", "currencyprimary", "Cost_base", "Qty", "Avg_price", "Current price", "unrealized pnl"]
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
                "Symbol":          st.column_config.TextColumn(),
                "Description":     st.column_config.TextColumn(),
                "currencyprimary": st.column_config.TextColumn("currencyprimary"),
                "Cost_base":       st.column_config.NumberColumn(format="%.2f"),
                "Qty":             st.column_config.NumberColumn(format="%.2f"),
                "Avg_price":       st.column_config.NumberColumn(format="%.2f"),
                "Current price":   st.column_config.NumberColumn(format="%.2f"),
                "unrealized pnl":  st.column_config.NumberColumn(format="%.2f"),
            }
        )


        # helper: show the net-qty pivot so you can verify the open/closed logic
        with st.expander("Show net quantity per symbol (pivot-style)"):
            st.dataframe(
                net_qty.rename(columns={sym_col: "Symbol"}),
                use_container_width=True,
                hide_index=True
            )

# ========================= PAGE: Option ROI Calculator =========================
elif page == "Option ROI Calculator":
    st.header("Option ROI Calculator")

    st.markdown("""
        <style>

        div[data-testid="stNumberInput"] label {
            color: #111827 !important;
            font-weight: 700 !important;
            font-size: 15px !important;
        }

        div[data-testid="stNumberInput"] > div {
            background-color: #fffbeb !important;
            border: 1px solid #fcd34d !important;
            border-radius: 10px !important;
            padding: 3px 8px !important;
        }

        div[data-testid="stNumberInput"] input {
            background-color: #fffbeb !important;
            color: #111827 !important;
            font-weight: 600 !important;
        }

        div[data-testid="stNumberInput"] button {
            background-color: #fffbeb !important;
            color: #111827 !important;
            border: none !important;
        }

        /* remove yellow underline */
        div[data-testid="stNumberInput"] div[data-baseweb="input"]::after {
            display: none !important;
        }

        </style>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 2])

    with col1:
        st.markdown('<div class="roi-wrap">', unsafe_allow_html=True)

        strike = st.number_input("Strike", min_value=0.0, value=40.0, step=1.00)
        premium = st.number_input("Premium", min_value=0.0, value=2.42, step=0.10)
        dte = st.number_input("DTE", min_value=1, value=132, step=1)

        capital = strike * 100
        roi = ((premium * 100) / capital) * 100 if capital > 0 else 0
        roi_annualized = roi * (365 / dte) if dte > 0 else 0

        # conditional formatting  ← SEM TO IDE
        if roi_annualized >= 30:
            bg_color = "#dcfce7"
            border_color = "#22c55e"
        else:
            bg_color = "#fffbeb"
            border_color = "#fcd34d"

        st.markdown("### Result")

        st.markdown(
            f"""
            <div style="
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 10px;
                padding: 12px 14px;
                margin-bottom: 12px;
                max-width: 320px;
            ">
                <div style="
                    font-size: 14px;
                    color: #111827;
                    font-weight: 600;
                    margin-bottom: 4px;
                ">ROI annualized</div>
                <div style="
                    font-size: 22px;
                    color: #111827;
                    font-weight: 700;
                ">{roi_annualized:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
# ========================= PAGE: Realized PnL Analysis =========================
elif page == "📊 Realized PnL Analysis":
    st.header("Realized PnL Analysis")

    df_rlz = load_realized(VIEW_REALIZED_FIFO_USD)

    if df_rlz.empty:
        st.info("No realized data available.")
        st.stop()

    # dates
    if "close_date" in df_rlz.columns:
        df_rlz["close_date"] = pd.to_datetime(df_rlz["close_date"], errors="coerce")
        df_rlz["year"] = df_rlz["close_date"].dt.year
        df_rlz["month"] = df_rlz["close_date"].dt.to_period("M").astype(str)
    else:
        st.error("Column 'close_date' is missing.")
        st.stop()

    # realized pnl usd column
    if "realized_pnl_usd_total" in df_rlz.columns:
        realized_col = "realized_pnl_usd_total"
    elif "realized_usd" in df_rlz.columns:
        realized_col = "realized_usd"
    else:
        st.error("No USD realized PnL column found.")
        st.stop()

    df_rlz[realized_col] = pd.to_numeric(df_rlz[realized_col], errors="coerce").fillna(0)
    df_rlz["pnl_type"] = df_rlz.apply(
        lambda r: f"{r['asset_class']}_POS" if r[realized_col] >= 0 else f"{r['asset_class']}_NEG",
        axis=1
    )

    color_scale = alt.Scale(
    domain=["OPT_POS","STK_POS","OPT_NEG","STK_NEG"],
    range=[
        "#16a34a",  # OPT positive (strong green)
        "#86efac",  # STK positive (light green)
        "#dc2626",  # OPT negative (strong red)
        "#fca5a5"   # STK negative (light red)
        ]
    )
    #color_scale = alt.Scale(
    #    domain=["OPT", "STK"],
    #    range=["#1565c0", "#7cb5e3"]
    #)

    left_col, right_col = st.columns(2)

    # =========================================================
    # LEFT SIDE = YEAR
    # =========================================================
    with left_col:
        st.subheader("Realized PnL in USD by Year")

        ticker_options_y = sorted(df_rlz["ticker"].dropna().astype(str).unique()) if "ticker" in df_rlz.columns else []
        selected_tickers_y = st.multiselect(
            "Ticker",
            options=ticker_options_y,
            key="ticker_year"
        )

        asset_options_y = sorted(df_rlz["asset_class"].dropna().astype(str).unique()) if "asset_class" in df_rlz.columns else []
        selected_asset_y = st.selectbox(
            "Asset class",
            options=["All"] + asset_options_y,
            key="asset_year"
        )

        year_options_y = sorted(df_rlz["year"].dropna().astype(int).unique())
        selected_years_y = st.multiselect(
            "Year",
            options=year_options_y,
            default=year_options_y,
            key="year_filter_year_chart"
        )

        df_year = df_rlz.copy()

        if selected_tickers_y and "ticker" in df_year.columns:
            df_year = df_year[df_year["ticker"].isin(selected_tickers_y)]

        if selected_asset_y != "All" and "asset_class" in df_year.columns:
            df_year = df_year[df_year["asset_class"] == selected_asset_y]

        if selected_years_y:
            df_year = df_year[df_year["year"].isin(selected_years_y)]

        st.markdown("---")

        if df_year.empty:
            st.info("No data for selected filters.")
        else:
            chart_year_df = (
                df_year
                .groupby(["year", "asset_class", "pnl_type"], as_index=False)[realized_col]
                .sum()
                .rename(columns={realized_col: "realized_pnl_usd"})
            )

            chart_year = alt.Chart(chart_year_df).mark_bar().encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("realized_pnl_usd:Q", title="Realized PnL in USD"),
                #color=alt.Color("asset_class:N", title="Asset class", scale=color_scale),
                color=alt.Color("pnl_type:N",scale=color_scale,legend=None),
                tooltip=[
                    alt.Tooltip("year:O", title="Year"),   # pri month grafe zmen na month
                    alt.Tooltip("asset_class:N", title="Asset class"),
                    alt.Tooltip("pnl_type:N", title="PnL type"),
                    alt.Tooltip("realized_pnl_usd:Q", title="Realized PnL USD", format=",.2f")
                ]
            ).properties(
                height=350
            )

            st.altair_chart(chart_year, use_container_width=True)

    # =========================================================
    # RIGHT SIDE = MONTH
    # =========================================================
    with right_col:
        st.subheader("Realized PnL in USD by Month")

        ticker_options_m = sorted(df_rlz["ticker"].dropna().astype(str).unique()) if "ticker" in df_rlz.columns else []
        selected_tickers_m = st.multiselect(
            "Ticker",
            options=ticker_options_m,
            key="ticker_month"
        )

        asset_options_m = sorted(df_rlz["asset_class"].dropna().astype(str).unique()) if "asset_class" in df_rlz.columns else []
        selected_asset_m = st.selectbox(
            "Asset class",
            options=["All"] + asset_options_m,
            key="asset_month"
        )

        year_options_m = sorted(df_rlz["year"].dropna().astype(int).unique())
        selected_years_m = st.multiselect(
            "Year",
            options=year_options_m,
            default=year_options_m,
            key="year_filter_month_chart"
        )

        df_month = df_rlz.copy()

        if selected_tickers_m and "ticker" in df_month.columns:
            df_month = df_month[df_month["ticker"].isin(selected_tickers_m)]

        if selected_asset_m != "All" and "asset_class" in df_month.columns:
            df_month = df_month[df_month["asset_class"] == selected_asset_m]

        if selected_years_m:
            df_month = df_month[df_month["year"].isin(selected_years_m)]

        st.markdown("---")

        if df_month.empty:
            st.info("No data for selected filters.")
        else:
            chart_month_df = (
                df_month
                .groupby(["month", "asset_class", "pnl_type"], as_index=False)[realized_col]
                .sum()
                .rename(columns={realized_col: "realized_pnl_usd"})
                .sort_values("month")
            )

            month_order = sorted(chart_month_df["month"].unique().tolist())

            chart_month = alt.Chart(chart_month_df).mark_bar().encode(
                x=alt.X("month:O", title="Month", sort=month_order),
                y=alt.Y("realized_pnl_usd:Q", title="Realized PnL in USD"),
                #color=alt.Color("asset_class:N", title="Asset class", scale=color_scale),
                color=alt.Color("pnl_type:N",scale=color_scale,legend=None),
                tooltip=[
                    alt.Tooltip("month:O", title="Month"),
                    alt.Tooltip("asset_class:N", title="Asset class"),
                    alt.Tooltip("pnl_type:N", title="PnL type"),
                    alt.Tooltip("realized_pnl_usd:Q", title="Realized PnL USD", format=",.2f")
                ]
            ).properties(
                height=350
            )

            st.altair_chart(chart_month, use_container_width=True)


# ========================= PAGE: Settings =========================
else:
    st.header("Settings")
    st.info("Configuration options will be here.")
