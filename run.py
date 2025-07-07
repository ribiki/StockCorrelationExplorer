"""Pipeline runner
================
Build price matrix → compute rolling correlations → launch Streamlit GUI.

Usage
-----
$ python run.py
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.correlation_engine import CorrelationEngine
from src.file_loader import build_price_matrix
from utils.config import CONFIG

logger = logging.getLogger(__name__)


# ───────────────────────#
# helper functions       #
# ───────────────────────#

def _compute_returns(price_df: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    1.  Drop any trading day that still has a NaN for *any* ticker.
    2.  Calculate % returns (the first row per ticker is NaN and is dropped)
    """
    clean = price_df.dropna(how="any")
    return clean.pct_change().iloc[1:].values.astype(np.float32), clean.index[1:]


def _stream_correlations(returns: np.ndarray, mmap_path: Path) -> None:
    """Feed daily returns through :class:`CorrelationEngine` and persist output."""
    n_days, n_tickers = returns.shape

    engine = CorrelationEngine(
        mmap_file_path=str(mmap_path),
        num_tickers=n_tickers,
        window_size=CONFIG.CORR_WINDOW,
        total_days=n_days,
    )

    for day, ret in enumerate(returns, 1):
        if day % 100 == 0 or day == n_days:
            logger.info("processing day %s / %s", day, n_days)
        engine.update(ret)
    engine.finalize()


# ───────────────────────#
# main function          #
# ───────────────────────#
def main() -> None:
    """Run the full pipeline and launch the Streamlit GUI."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")

    base_dir = Path(__file__).parent.resolve()
    raw_zip = base_dir / "data" / "raw" / "stock_data.zip"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    mmap_path = processed_dir / "corr_mmap.bin"

    # 1. build price matrix --------------
    price_df = build_price_matrix(raw_zip)
    logger.info("price matrix shape: %s", price_df.shape)

    if len(price_df) < CONFIG.CORR_WINDOW + 1:
        sys.exit("Not enough rows to compute rolling correlations")

    # 2. daily returns -------------------
    returns, valid_dates = _compute_returns(price_df)
    logger.info("computed %s daily‑return rows", len(returns))

    # 3. rolling correlations -------------
    logger.info("computing rolling correlations …")
    _stream_correlations(returns, mmap_path)
    logger.info("correlation mmap written to %s", mmap_path)

    # 4. record metadata --------------------
    meta = {
        "start_date": price_df.index.min().strftime("%Y-%m-%d"),
        "end_date": price_df.index.max().strftime("%Y-%m-%d"),
    }
    meta_path = processed_dir / "latest_run_dates.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("saved metadata → %s", meta_path)

    # 5. launch GUI ----------------------------
    logger.info("starting Streamlit GUI …")
    gui_script = base_dir / "app" / "run_gui_app.py"
    subprocess.run(["streamlit", "run", str(gui_script)])


if __name__ == "__main__":
    main()
