import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import os

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from utils.config import CONFIG


@st.cache_resource(ttl=3600)  # Refresh cache hourly
def load_data():
    try:
        proc_dir = Path(CONFIG.PROCESSED_DATA_DIR)
        if not proc_dir.is_absolute():
            proc_dir = project_root / proc_dir

        proc_dir.mkdir(parents=True, exist_ok=True)
        price_files = list(proc_dir.glob("price_matrix_*.parquet"))

        if not price_files:
            st.error("❌ No price data files found")
            return None, None, None, None

        dfs = []
        for f in price_files:
            try:
                df = pd.read_parquet(f)
                if "Date" in df.columns:
                    df = df.rename(columns={"Date": "date"})
                if "date" not in df.columns:
                    st.error(f"⚠️ File {f.name} missing 'date' column")
                    continue
                df = df.set_index("date")
                dfs.append(df)
            except Exception as e:
                st.error(f"⚠️ Error loading {f.name}: {e}")

        if not dfs:
            return None, None, None, None

        price_df = pd.concat(dfs).sort_index()
        price_df = price_df[~price_df.index.duplicated(keep='last')]

        min_date = price_df.index.min()
        max_date = price_df.index.max()
        tickers = price_df.columns.tolist()

        return price_df, tickers, min_date, max_date
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {str(e)}")
        return None, None, None, None


def calculate_correlation(price_df, t1, t2, start_date, end_date):
    """
    Calculate correlation and statistics for two tickers given a date range
    """
    try:
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        # Validate tickers exist in data
        if t1 not in price_df.columns:
            st.error(f"⚠️ Ticker '{t1}' not found in price data columns")
            return None, None, None, None
        if t2 not in price_df.columns:
            st.error(f"⚠️ Ticker '{t2}' not found in price data columns")
            return None, None, None, None

        # Filter data for selected date range
        mask = (price_df.index >= start_date) & (price_df.index <= end_date)
        period_df = price_df.loc[mask, [t1, t2]]

        st.info(f"Date range selected: {start_date.date()} to {end_date.date()}")
        st.info(f"Total days in range: {len(period_df)}")
        st.info(f"Days with {t1} data: {period_df[t1].notna().sum()}")
        st.info(f"Days with {t2} data: {period_df[t2].notna().sum()}")

        # Drop rows with missing values
        period_df = period_df.dropna()

        if len(period_df) == 0:
            st.error("❌ No overlapping data points for both tickers in selected date range")
            return None, None, None, None

        if len(period_df) < 10:
            st.warning(f"⚠️ Only {len(period_df)} common trading days available. Minimum 10 required.")

            # Show missing dates analysis
            missing_dates = period_df.index.to_series().diff().dt.days
            gaps = missing_dates[missing_dates > 1]
            if not gaps.empty:
                st.warning(f"Data gaps detected: {gaps.value_counts().to_dict()}")

            return None, None, None, None

        returns = period_df.pct_change().dropna()
        correlation = returns.corr().iloc[0, 1]

        # Calculate statistics for each ticker
        stats = {}
        for ticker in [t1, t2]:
            prices = period_df[ticker]

            total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

            daily_std = returns[ticker].std() * 100
            monthly_std = daily_std * np.sqrt(21)
            annualized_std = daily_std * np.sqrt(252)

            stats[ticker] = {
                "Total Return": total_return,
                "Daily Std Dev": daily_std,
                "Monthly Std Dev": monthly_std,
                "Annualized Std Dev": annualized_std
            }

        return correlation, stats, period_df.index[0], period_df.index[-1]

    except Exception as e:
        st.error(f"⚠️ Error in correlation calculation: {str(e)}")
        st.error(traceback.format_exc())
        return None, None, None, None

def main():
    st.set_page_config(layout="wide")
    st.title("🔗 Stock Correlation Explorer")

    price_df, tickers, min_date, max_date = load_data()
    if price_df is None:
        return

    st.subheader("Data Information")
    st.info(f"**Loaded Data Range:** {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    st.info(f"**Tickers Available:** {len(tickers)} symbols")

    st.subheader("Analysis Period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date:",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
    with col2:
        end_date = st.date_input(
            "End Date:",
            value=max_date,
            min_value=start_date,
            max_value=max_date
        )

    st.subheader("Ticker Selection")
    t1, t2 = st.columns(2)
    with t1:
        ticker1 = st.selectbox("Select First Ticker:", tickers, index=0)
    with t2:
        default_idx = 1 if len(tickers) > 1 and tickers[1] != ticker1 else min(1, len(tickers) - 1)
        ticker2 = st.selectbox("Select Second Ticker:", tickers, index=default_idx)

    if st.button("Calculate Correlation", type="primary"):
        if ticker1 == ticker2:
            st.error("❌ Please select two different tickers")
            return

        with st.spinner("Calculating..."):
            correlation, stats, actual_start, actual_end = calculate_correlation(
                price_df, ticker1, ticker2, start_date, end_date
            )

            if correlation is None:
                return

            # Display results
            st.success(f"Correlation Coefficient: **{correlation:.4f}**")
            st.caption(f"Period: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}")

            # Create metrics columns
            col1, col2 = st.columns(2)
            metric_data = []
            for i, ticker in enumerate([ticker1, ticker2]):
                col = col1 if i == 0 else col2
                with col:
                    st.subheader(f"{ticker} Performance")
                    if ticker in stats:
                        st.metric("Total Return", f"{stats[ticker]['Total Return']:.2f}%")
                        st.metric("Annual Volatility", f"{stats[ticker]['Annualized Std Dev']:.2f}%")
                    else:
                        st.warning("No data available")

                    # Add to table data
                    if ticker in stats:
                        metric_data.append({
                            "Ticker": ticker,
                            "Correlation": correlation,
                            "Total Return (%)": stats[ticker]["Total Return"],
                            "Annual Volatility (%)": stats[ticker]["Annualized Std Dev"]
                        })

            if metric_data:
                st.subheader("Performance Summary")
                df = pd.DataFrame(metric_data)
                st.dataframe(df.style.format({
                    "Correlation": "{:.4f}",
                    "Total Return (%)": "{:.2f}%",
                    "Annual Volatility (%)": "{:.2f}%"
                }))


if __name__ == "__main__":
    main()
