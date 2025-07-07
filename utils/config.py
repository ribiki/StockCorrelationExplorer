"""
Global configuration
====================

*Single source of truth* for paths and tunables used across the pipeline.
Edit values here — the rest of the code imports :data:`CONFIG` and stays clean.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from utils import config as cfg_mod

@dataclass(slots=True, frozen=True)
class _Config:
    # ── data locations ──────────────────────────────────────────────────
    #
    # All paths are *relative to the project root*

    RAW_ZIP: Path = Path("data/raw/stock_data.zip")
    PROCESSED_DATA_DIR: Path = Path("data/processed")

    # ── cleaning thresholds ────────────────────────────────────────────
    #
    # If more than 35 % of the rows for a ticker are missing after the
    # forward/back fill step, that ticker is dropped.

    MAX_STOCK_MISSING_PCT: float = 0.35   # 0.0 – 1.0

    # ── correlation engine ─────────────────────────────────────────────
    #
    # Window length in *trading days* for the rolling Pearson correlation.

    CORR_WINDOW: int = 20

    # ── calendar settings (optional)────────────────────────────────────
    #
    # Only needed if you later add a “get_trading_days” utility.

    MARKET_CALENDAR: str = "NYSE"


# instance imported elsewhere
CONFIG = _Config()
