import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from utils.config import CONFIG
from src.correlation_engine import CorrelationEngine


@st.cache_resource(ttl=3600)
def load_price_data() -> Tuple[pd.DataFrame, list[str]]:
    """
    Return
    ------
    price_df: pd.DataFrame(NaNs value will remain only for long gaps, e.g. 20200909 - 20201231)
    """
    proc_dir = (project_root / CONFIG.PROCESSED_DATA_DIR).resolve()
    parquet_files = sorted(proc_dir.glob("price_matrix_*.parquet"))

    if not parquet_files:
        st.error("❌ No Parquet price files found in data/processed")
        return pd.DataFrame(), []

    # concatenate all price matrices found
    frames = []
    for fp in parquet_files:
        df = pd.read_parquet(fp)
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        if "date" not in df.columns:
            st.warning(f"{fp.name}: no 'date' column")
            continue
        frames.append(df.set_index("date"))

    if not frames:
        return pd.DataFrame(), []
    price_df = pd.concat(frames).sort_index()
    price_df = price_df.loc[~price_df.index.duplicated(keep="last")]

    return price_df, price_df.columns.tolist()


# ────────────────────────────────────────#
# 2.  Memory‑mapped rolling correlations  #
# ────────────────────────────────────────#
# This is cached with no expriry
@st.cache_resource
def load_corr_engine() -> CorrelationEngine:
    path = Path(CONFIG.PROCESSED_DATA_DIR) / "corr_mmap.bin"
    return CorrelationEngine(path)


# ───────────────────── #
# 3.  Streamlit GUI     #
# ───────────────────── #
def main() -> None:
    st.set_page_config(layout="wide")
    st.title("🔗 Stock Correlation Explorer")

    # ---------- load data ---------------
    price_df, tickers = load_price_data()
    if price_df.empty:
        return

    engine = load_corr_engine()

    # rebuild the calendar the pipeline used (rows without NaN)
    clean_df = price_df.dropna(how="any")
    if clean_df.empty:
        st.error("All rows contain at least one NaN value.")
        return

    returns_index = clean_df.index[1:]                              # after pct_change
    corr_dates = returns_index[CONFIG.CORR_WINDOW - 1:]             # first valid window

    st.subheader("Data Information")
    st.info(f"**Loaded price range:** {price_df.index.min().date()} ➜ {price_df.index.max().date()}")
    st.info(f"**Correlations available for:** {corr_dates[0].date()} ➜ {corr_dates[-1].date()}")
    st.info(f"**Tickers:** {len(tickers)} symbols")

    # ==========================================#
    # SECTION A · Daily correlation explorer    #
    # ==========================================#
    st.markdown("---")
    st.subheader("🔍 Explore correlations on **a trading day**")

    c1, c2 = st.columns([2, 1])
    with c1:
        chosen_date = st.select_slider(
            "Trading day:",
            options=list(corr_dates),
            value=corr_dates[-1],
            format_func=lambda d: d.strftime("%Y‑%m‑%d"),
        )
        day_idx = corr_dates.get_indexer([chosen_date])[0]

    with c2:
        ref_ticker = st.selectbox("Reference ticker:", tickers, index=0)

    if st.button("Show daily correlations", type="primary"):
        with st.spinner("Fetching correlation matrix…"):
            mat = engine.get_day_matrix(day_idx)
            ref_vec = pd.Series(mat[tickers.index(ref_ticker)], index=tickers)

            # top / bottom 10 correlations
            top10 = ref_vec.drop(ref_ticker).nlargest(10).rename("ρ(+)")
            bot10 = ref_vec.nsmallest(10).rename("ρ(–)")

            st.write(f"### {ref_ticker} – {chosen_date.date()}")
            st.write("Top 10 positively / negatively correlated tickers")

            tbl = pd.concat([top10, bot10], axis=1).style.format("{:.3f}")
            st.dataframe(tbl, height=320)

            # quick 50×50 heat‑map of strongest absolute correlations
            top50 = ref_vec.abs().nlargest(50).index
            heat_df = pd.DataFrame(mat, index=tickers, columns=tickers).loc[top50, top50]
            fig = px.imshow(
                heat_df,
                title=f"Top‑50 correlation sub‑matrix on {chosen_date.strftime('%Y‑%m‑%d')}",
                aspect="auto",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ====================================#
    # SECTION B · Pair history explorer   #
    # ====================================#
    st.markdown("---")
    st.subheader("📈 Correlation history for *two* tickers")

    p1, p2 = st.columns(2)
    with p1:
        tkr1 = st.selectbox("Ticker A:", tickers, index=0, key="tkr1")
    with p2:
        default_idx = 1 if len(tickers) > 1 and tickers[1] != tkr1 else min(1, len(tickers) - 1)
        tkr2 = st.selectbox("Ticker B:", tickers, index=default_idx, key="tkr2")

    d1, d2 = st.columns(2)
    with d1:
        start_dt = st.date_input("Start date:", value=corr_dates[0].date(),
                                 min_value=corr_dates[0].date(),
                                 max_value=corr_dates[-1].date())
    with d2:
        end_dt = st.date_input("End date:", value=corr_dates[-1].date(),
                               min_value=start_dt,
                               max_value=corr_dates[-1].date())

    if st.button("Show history", type="primary", key="history_btn"):
        if tkr1 == tkr2:
            st.error("❌ Please choose *different* tickers")
        else:
            with st.spinner("Loading pair history…"):
                i, j = tickers.index(tkr1), tickers.index(tkr2)
                hist_arr = engine.get_pair_history(i, j)
                if hist_arr.size == len(corr_dates):
                    date_idx = corr_dates  # perfect match
                elif hist_arr.size < len(corr_dates):
                    # mmap is shorter - take the last N dates so shapes match
                    date_idx = corr_dates[-hist_arr.size:]
                else:
                    # mmap is longer - fallback to a trimmed array (should not happen)
                    hist_arr = hist_arr[-len(corr_dates):]
                    date_idx = corr_dates

                hist = pd.Series(hist_arr, index=date_idx, name="ρ (20‑day)")

                rng_mask = (hist.index >= pd.Timestamp(start_dt)) & (hist.index <= pd.Timestamp(end_dt))
                sub = hist.loc[rng_mask]

                if sub.empty:
                    st.warning("No correlation values in the selected date range "
                               "(likely because some days had missing prices).")
                else:
                    latest = sub.iloc[-1]
                    st.success(f"Latest correlation ({sub.index[-1].date()}): **{latest:.4f}**")
                    st.line_chart(sub)

                    # quick stats
                    st.caption(f"Mean ρ: {sub.mean():.4f}·Stddev: {sub.std():.4f} "
                               f"·Min: {sub.min():.4f}·Max: {sub.max():.4f}")

    # ============================================#
    # SECTION C · Price‑based performance metrics #
    # ============================================#
    st.markdown("---")
    st.subheader("🚀 Price‑performance snapshot")

    if st.checkbox("Show basic return & volatility metrics"):
        # dataframe has short gap NaNs only
        def perf_stats(tkr: str):
            prices = price_df[tkr].dropna()
            rets = prices.pct_change().dropna()
            return {
                "Total return%": (prices.iloc[-1] / prices.iloc[0] - 1) * 100,
                "Ann. vol%": rets.std() * np.sqrt(252) * 100,
                "Obs. days": len(prices)
            }

        data = pd.DataFrame({t: perf_stats(t) for t in [tkr1, tkr2]}).T
        st.dataframe(data.style.format("{:.2f}"))

    st.caption("Streamlit demo – correlations pre‑computed with a rolling window"
               f"of {CONFIG.CORR_WINDOW} trading days.")


if __name__ == "__main__":
    main()