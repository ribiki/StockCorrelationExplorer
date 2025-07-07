from __future__ import annotations

import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import List, Tuple
from unittest import mock

import pandas as pd

from utils import config as config_mod


# ──────────────────────────────────────────────────────────────────────────────
# helper: convert a ZIP containing arbitrary CSV frames and keep temp dir alive
# ──────────────────────────────────────────────────────────────────────────────
def _make_zip(csv_frames: List[pd.DataFrame]) -> Tuple[Path, tempfile.TemporaryDirectory]:
    """
    Return (zip_path, temp_dir_obj).  Keep the TemporaryDirectory object alive
    """
    tmp_obj = tempfile.TemporaryDirectory()
    td = Path(tmp_obj.name)
    for i, frame in enumerate(csv_frames, start=1):
        (td / f"file_{i}.csv").write_text(frame.to_csv(index=False))
    zip_path = td / "sample.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for csv in td.glob("*.csv"):
            z.write(csv, arcname=csv.name)
    return zip_path, tmp_obj


# ──────────────────────────────────────────────────────────────────────────────
# test-case
# ──────────────────────────────────────────────────────────────────────────────
class FileLoaderTestCase(unittest.TestCase):
    """Validate CSV → cleanup → matrix pipeline plus edge-cases."""

    @classmethod
    def setUpClass(cls):
        cls.dates = pd.date_range("2024-01-02", periods=3, freq="B")
        cls.csv_aaa = pd.DataFrame(
            {"Ticker": ["AAA"] * 3, "Date": cls.dates, "Price": [10.0, 10.5, 11.0]}
        )
        cls.csv_bbb_full = pd.DataFrame(
            {"Ticker": ["BBB"] * 3, "Date": cls.dates, "Price": [20.0, 19.5, 21.0]}
        )
        cls.csv_bbb_sparse = pd.DataFrame(  # only first day
            {"Ticker": ["BBB"], "Date": [cls.dates[0]], "Price": [20.0]}
        )

    # --------------------------------------------------------------------- #
    def test_matrix_shape_and_sample_value(self):
        zip_path, td_obj = _make_zip([self.csv_aaa, self.csv_bbb_full])

        with tempfile.TemporaryDirectory() as proc_dir:
            new_cfg = config_mod.CONFIG.__class__(
                RAW_ZIP=config_mod.CONFIG.RAW_ZIP,
                PROCESSED_DATA_DIR=Path(proc_dir),
                MAX_STOCK_MISSING_PCT=config_mod.CONFIG.MAX_STOCK_MISSING_PCT,
                CORR_WINDOW=config_mod.CONFIG.CORR_WINDOW,
                MARKET_CALENDAR=config_mod.CONFIG.MARKET_CALENDAR,
            )
            with mock.patch("utils.config.CONFIG", new=new_cfg):
                import importlib, src.file_loader as fl
                importlib.reload(fl)

                df = fl.build_price_matrix(zip_path)

            self.assertListEqual(list(df.columns), ["AAA", "BBB"])
            self.assertEqual(len(df), 3)
            self.assertEqual(df.loc[self.dates[0], "AAA"], 10.0)
            # parquet now saved in proc_dir
            self.assertTrue(any(Path(proc_dir).glob("price_matrix_*.parquet")))

        td_obj.cleanup()

    # --------------------------------------------------------------------- #
    def test_sparse_ticker_kept_after_fill(self):
        zip_path, td_obj = _make_zip([self.csv_aaa, self.csv_bbb_sparse])

        strict_cfg = config_mod.CONFIG.__class__(
            RAW_ZIP=config_mod.CONFIG.RAW_ZIP,
            PROCESSED_DATA_DIR=Path(tempfile.mkdtemp()),
            MAX_STOCK_MISSING_PCT=0.50,
            CORR_WINDOW=config_mod.CONFIG.CORR_WINDOW,
            MARKET_CALENDAR=config_mod.CONFIG.MARKET_CALENDAR,
        )
        with mock.patch("utils.config.CONFIG", new=strict_cfg):
            import importlib, src.file_loader as fl
            importlib.reload(fl)
            df = fl.build_price_matrix(zip_path)

        self.assertListEqual(list(df.columns), ["AAA", "BBB"])
        shutil.rmtree(strict_cfg.PROCESSED_DATA_DIR, ignore_errors=True)
        td_obj.cleanup()

    # --------------------------------------------------------------------- #
    def test_missing_required_column_raises(self):
        """CSV lacking 'Price' column triggers ValueError."""
        bad_csv = pd.DataFrame({"Ticker": ["AAA"], "Date": [self.dates[0]]})
        zip_file, td_obj = _make_zip([bad_csv])

        with self.assertRaises(ValueError):
            from src.file_loader import build_price_matrix
            build_price_matrix(zip_file)

        td_obj.cleanup()

    # --------------------------------------------------------------------- #
    def test_fill_and_drop_logic(self):
        """
        _pivot_price_matrix should forward/back-fill gaps, keep BBB (0 % NaNs
        after fill), yield only the two dates present in input.
        """
        from src.file_loader import _pivot_price_matrix
        from utils import config as cfg_mod

        tidy_rows = [
            {"ticker": "AAA", "date": self.dates[0], "price": 11.0},
            {"ticker": "AAA", "date": self.dates[2], "price": 11.0},  # gap on day-1
            {"ticker": "BBB", "date": self.dates[0], "price": 20.0},  # single row
        ]
        tidy_df = pd.DataFrame(tidy_rows)

        strict_cfg = cfg_mod.CONFIG.__class__(
            RAW_ZIP            = cfg_mod.CONFIG.RAW_ZIP,
            PROCESSED_DATA_DIR = cfg_mod.CONFIG.PROCESSED_DATA_DIR,
            MAX_STOCK_MISSING_PCT = 0.50,
            CORR_WINDOW        = cfg_mod.CONFIG.CORR_WINDOW,
            MARKET_CALENDAR    = cfg_mod.CONFIG.MARKET_CALENDAR,
        )

        with mock.patch("utils.config.CONFIG", new=strict_cfg):
            matrix = _pivot_price_matrix(tidy_df)

        self.assertListEqual(list(matrix.columns), ["AAA", "BBB"])  # BBB kept
        self.assertEqual(len(matrix), 2)                            # two dates
        self.assertEqual(matrix.iloc[1]["AAA"], 11.0)               # ffill ok
        self.assertFalse(matrix.isna().values.any())                # no NaNs


if __name__ == "__main__":
    unittest.main(verbosity=2)
