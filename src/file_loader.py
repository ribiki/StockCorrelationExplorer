"""Utility helpers to load zipped CSV price data → Parquet matrix.

Usage (from run.py)::

    from src.file_loader import build_price_matrix
    df = build_price_matrix("data/raw/stock_data.zip")
"""
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas_market_calendars as mcal

from utils.config import CONFIG

# -----------------------------#
# low‑level helper functions   #
# -----------------------------#

REQ_COLS = {"Ticker", "Date", "Price"}


def _unzip_archives(source: str | os.PathLike, dest: Path) -> List[Path]:
    """Extract *all* ``.zip`` files found in *source* into *dest*.

    Parameters
    ----------
    source
        Either a single ``.zip`` file or a directory that *contains* zip files.
    dest
        Target directory. Created if missing.

    Returns
    -------
    list[Path]
        List of *CSV* files found inside the archives.
    """
    dest.mkdir(parents=True, exist_ok=True)

    def _extract_one(zf: Path) -> None:
        with zipfile.ZipFile(zf) as z:
            z.extractall(dest)

    src_path = Path(source)
    if src_path.is_dir():
        zips = list(src_path.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"No .zip files inside directory {source!r}.")
        for zf in zips:
            _extract_one(zf)
    elif src_path.suffix.lower() == ".zip":
        _extract_one(src_path)
    else:
        raise ValueError("source must be a .zip file or directory containing .zip files")

    return list(dest.rglob("*.csv"))


def _read_csv_files(csv_files: Iterable[Path]) -> pd.DataFrame:
    """Load every CSV into one tidy dataframe (ticker, date, price)."""
    records: list[pd.DataFrame] = []
    for fp in tqdm(csv_files, desc="Reading CSVs"):
        df = pd.read_csv(fp)
        missing = REQ_COLS - set(df.columns)
        if missing:
            raise ValueError(f"CSV {fp.name} missing columns: {', '.join(missing)}")
        df = df[list(REQ_COLS)].rename(columns={"Ticker": "ticker", "Date": "date", "Price": "price"})
        records.append(df)

    if not records:
        raise RuntimeError("No CSV files could be read – empty list?")

    all_rows = pd.concat(records, ignore_index=True)
    all_rows["date"] = pd.to_datetime(all_rows["date"], errors="coerce")
    all_rows.dropna(subset=["date", "ticker", "price"], inplace=True)
    return all_rows


def _pivot_price_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert tidy frame into a [date × ticker] price matrix,
    re‑index to a full NYSE calendar and *gap‑limited* fill.

    Long gaps (>=4 trading days) remain NaN so they can later be skipped by the
    correlation engine.
    """
    matrix = (
        raw.pivot_table(index="date", columns="ticker", values="price", aggfunc="last")
        .sort_index()
        .astype(np.float32)
    )

    #  1)  use trading calendar
    cal = mcal.get_calendar(CONFIG.MARKET_CALENDAR)
    full_days = cal.schedule(
        start_date=matrix.index.min(),
        end_date=matrix.index.max()
    ).index
    matrix = matrix.reindex(full_days)

    # 2)  only fill <= 3 consecutive sessions
    matrix = matrix.ffill(limit=3).bfill(limit=3)

    # 3)  drop inactive tickers
    too_sparse = matrix.isna().mean() > CONFIG.MAX_STOCK_MISSING_PCT
    if too_sparse.any():
        n = int(too_sparse.sum())
        print(f"Dropping {n} tickers (> {CONFIG.MAX_STOCK_MISSING_PCT:.0%} NaN)")
        matrix = matrix.loc[:, ~too_sparse]

    return matrix

def _save_to_parquet(df: pd.DataFrame) -> Path:
    out_dir = Path(CONFIG.PROCESSED_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / (
        f"price_matrix_{df.index.min():%Y%m%d}_{df.index.max():%Y%m%d}.parquet"
    )
    df.reset_index(names="date").to_parquet(fname)
    print(f"💾 Saved price matrix → {fname}")
    return fname


# -----------------------------#
# Public functions             #
# -----------------------------#
def build_price_matrix(zip_file_path: str | os.PathLike) -> pd.DataFrame:
    """Main entry‑point: zipped CSVs ⇒ clean price matrix (DataFrame).

    1. Extract all *.zip* archives               → temporary dir
    2. Load & concatenate CSV rows               → tidy frame (ticker, date, price)
    3. Pivot tidy frame to [date × ticker]       → price matrix
    4. Forward/back‑fill, drop sparse tickers    → cleaned matrix
    5. Save as Parquet (for GUI & downstream)    → *data/processed* directory
    """
    print(f" Building price matrix from {zip_file_path}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        csvs = _unzip_archives(zip_file_path, Path(tmp_dir))
        if not csvs:
            raise FileNotFoundError("No CSV files inside provided archive(s)")

        tidy = _read_csv_files(csvs)
        price_df = _pivot_price_matrix(tidy)
        _save_to_parquet(price_df)
        return price_df