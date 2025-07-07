from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from unittest import mock

from src.correlation_engine import CorrelationEngine
from utils.config import CONFIG


def _reference_roll_corr(returns: np.ndarray, window: int) -> np.ndarray:
    """Upper-triangular rolling correlation, computed with Pandas (for truth)."""
    n_days, n_tickers = returns.shape
    n_pairs = n_tickers * (n_tickers - 1) // 2
    out = np.empty((n_days - window + 1, n_pairs), dtype=np.float32)

    k = 0
    for i in range(n_tickers):
        for j in range(i + 1, n_tickers):
            s1, s2 = map(pd.Series, (returns[:, i], returns[:, j]))
            out[:, k] = s1.rolling(window).corr(s2)[window - 1 :].values
            k += 1
    return out


class CorrelationEngineTestCase(unittest.TestCase):
    """Compare engine output to NumPy/Pandas reference."""

    def setUp(self):
        np.random.seed(42)  # deterministic dataset for every test

    # ── core numerical agreement test ────────────────────────────────────────
    def test_engine_matches_reference(self):
        n_days, n_tickers, window = 30, 4, 5
        returns = np.random.normal(0, 0.01, size=(n_days, n_tickers)).astype(
            np.float32
        )

        with tempfile.TemporaryDirectory() as td:
            mmap_path = Path(td) / "corr.bin"

            # write mode
            writer = CorrelationEngine(
                mmap_file_path=str(mmap_path),
                num_tickers=n_tickers,
                window_size=window,
                total_days=n_days,
            )
            for row in returns:
                writer.update(row)
            writer.finalize()

            # read mode
            reader = CorrelationEngine(mmap_file_path=str(mmap_path))
            self.assertEqual(
                reader.num_corr_days, n_days - window + 1, "unexpected #corr windows"
            )

            ref_triu = _reference_roll_corr(returns, window)

            # compare first & last window only (speed)
            for day_idx in (0, reader.num_corr_days - 1):
                eng_mat = reader.get_day_matrix(day_idx)

                # build full matrix from reference triu for comparison
                full = np.eye(n_tickers, dtype=np.float32)
                full[np.triu_indices(n_tickers, k=1)] = ref_triu[day_idx]
                full += full.T - np.eye(n_tickers)

                self.assertTrue(
                    np.allclose(eng_mat, full, rtol=1e-3, atol=1e-3),
                    f"mismatch on day {day_idx}",
                )

    def test_window_length_from_config(self):
        # Build a new Config identical to the global one but with a tiny window
        new_cfg = CONFIG.__class__(
            RAW_ZIP=CONFIG.RAW_ZIP,
            PROCESSED_DATA_DIR=CONFIG.PROCESSED_DATA_DIR,
            MAX_STOCK_MISSING_PCT=CONFIG.MAX_STOCK_MISSING_PCT,
            CORR_WINDOW=3,
            MARKET_CALENDAR=CONFIG.MARKET_CALENDAR,
        )

        with mock.patch("utils.config.CONFIG", new=new_cfg):
            n_days, n_tickers = 5, 3
            returns = np.random.randn(n_days, n_tickers).astype(np.float32)

            with tempfile.TemporaryDirectory() as td:
                mmap_path = Path(td) / "corr.bin"
                writer = CorrelationEngine(
                    mmap_file_path=str(mmap_path),
                    num_tickers=n_tickers,
                    window_size=new_cfg.CORR_WINDOW,
                    total_days=n_days,
                )
                for row in returns:
                    writer.update(row)
                writer.finalize()

                reader = CorrelationEngine(mmap_file_path=str(mmap_path))
                self.assertEqual(reader.window_size, 3)
                self.assertEqual(reader.num_corr_days, n_days - 3 + 1)
