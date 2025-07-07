import mmap
import os
from pathlib import Path
from typing import Optional

import numpy as np
from numba import jit, prange
from dataclasses import dataclass

class CorrelationEngine:
    """
    rolling‐window correlation writer/reader using a memory‑mapped file.

    *Write mode*
    -------------
    >>> engine = CorrelationEngine("corr.bin", num_tickers=500, window_size=90, total_days=252*5)
    >>> for day_returns in returns:
    ...     engine.update(day_returns)
    >>> engine.finalize()

    *Read mode*
    -----------
    >>> engine = CorrelationEngine("corr.bin")
    >>> mat  = engine.get_day_matrix(0)   # correlation matrix of first window
    >>> hist = engine.get_pair_history(10, 42)  # float32 array, one value per window
    """

    #: bytes in the fixed header  (4 × int32)
    _HEADER_BYTES = 16
    #: bytes per stored correlation (float16)
    _ITEM_BYTES = 2

    # ───────────────────────#
    # public function        #
    # ───────────────────────#

    def __init__(
        self,
        mmap_file_path: str | os.PathLike,
        num_tickers: Optional[int] = None,
        window_size: Optional[int] = None,
        total_days: Optional[int] = None,
    ) -> None:
        self.mmap_file_path = Path(mmap_file_path)

        if num_tickers and window_size and total_days:  # write mode
            self._init_write(num_tickers, window_size, total_days)
        else:  # read‑only mode
            self._init_read()

    # ─────────────────────#
    # write mode helpers   #
    # ─────────────────────#

    def _init_write(self, n: int, w: int, total_days: int) -> None:
        self.num_tickers   = n
        self.window_size   = w
        self.total_days    = total_days
        self.num_pairs     = n * (n - 1) // 2
        self.max_windows   = max(0, total_days - w + 1)
        self.days_written  = 0
        self.num_corr_days = 0  # will be set in ``finalize``

        self._buffer       = np.zeros((w, n), dtype=np.float32)
        self._sum_r        = np.zeros(n,            dtype=np.float64)
        self._sum_r2       = np.zeros(n,            dtype=np.float64)
        self._sum_cross    = np.zeros(self.num_pairs, dtype=np.float64)

        self._i_idx, self._j_idx = self._build_index_arrays(n)
        self._create_mmap_file()

    def _create_mmap_file(self) -> None:
        required_bytes = self._HEADER_BYTES + self.max_windows * self.num_pairs * self._ITEM_BYTES
        with open(self.mmap_file_path, "wb") as f:
            header = np.array([self.num_tickers, self.window_size, 0, self._HEADER_BYTES], dtype=np.int32)
            f.write(header.tobytes())
            f.truncate(required_bytes)

        self._file_handle = open(self.mmap_file_path, "r+b")
        self._mmap = mmap.mmap(self._file_handle.fileno(), 0)
        self._data_offset = self._HEADER_BYTES

    # ───────────────────────#
    # read mode helpers      #
    # ───────────────────────#

    def _init_read(self) -> None:
        with open(self.mmap_file_path, "rb") as f:
            header = np.frombuffer(f.read(self._HEADER_BYTES), dtype=np.int32)

        self.num_tickers, self.window_size, self.num_corr_days, self._data_offset = header.tolist()  # type: ignore
        self.num_pairs = self.num_tickers * (self.num_tickers - 1) // 2
        self._i_idx, self._j_idx = self._build_index_arrays(self.num_tickers)

        self._file_handle = open(self.mmap_file_path, "rb")
        self._mmap = mmap.mmap(self._file_handle.fileno(), 0, access=mmap.ACCESS_READ)

    # ────────────────────────────────#
    # Numba‑accelerated              #
    # ────────────────────────────────#

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_corr(sum_cross, sum_r, sum_r2, i_idx, j_idx, w, n_pairs):
        out = np.empty(n_pairs, dtype=np.float32)
        for k in prange(n_pairs):
            i, j = i_idx[k], j_idx[k]
            cov   = sum_cross[k] - (sum_r[i] * sum_r[j] / w)
            var_i = sum_r2[i]   - (sum_r[i] ** 2 / w)
            var_j = sum_r2[j]   - (sum_r[j] ** 2 / w)
            if var_i < 1e-8 or var_j < 1e-8:
                out[k] = np.nan
            else:
                out[k] = cov / np.sqrt(var_i * var_j)
        return out

    # ──────────────────────#
    # public – write mode   #
    # ──────────────────────#

    def update(self, daily_returns: np.ndarray) -> None:
        """Add one new vector of *daily* returns (float32, length = num_tickers)."""
        if not hasattr(self, "_buffer"):
            raise RuntimeError("Engine opened in read‑only mode; cannot call update().")

        slot = self.days_written % self.window_size
        old  = self._buffer[slot].copy()
        self._buffer[slot] = daily_returns

        # incremental sums (add new, subtract old exiting the window)
        self._sum_r      += daily_returns - old
        self._sum_r2     += daily_returns ** 2 - old ** 2
        delta_outer       = np.outer(daily_returns, daily_returns) - np.outer(old, old)
        self._sum_cross  += delta_outer[self._i_idx, self._j_idx]

        self.days_written += 1
        if self.days_written >= self.window_size:
            self._write_window()

    def finalize(self) -> None:
        """Flush header with *actual* number of correlation windows and close the file."""
        if hasattr(self, "_buffer"):
            self.num_corr_days = max(0, self.days_written - self.window_size + 1)
            self._mmap.seek(8)  # header[2]
            self._mmap.write(np.int32(self.num_corr_days).tobytes())
            self._mmap.flush()

        self._mmap.close()
        self._file_handle.close()

    # ─────────────────────────────#
    #  read mode helpers           #
    # ─────────────────────────────#

    def get_day_matrix(self, day_index: int) -> np.ndarray:
        """Return the full correlation matrix (float32) for *day_index*."""
        if day_index < 0 or day_index >= self.num_corr_days:
            return np.eye(self.num_tickers, dtype=np.float32)

        start = self._data_offset + day_index * self.num_pairs * self._ITEM_BYTES
        buffer = self._mmap[start : start + self.num_pairs * self._ITEM_BYTES]
        triu_vals = np.frombuffer(buffer, dtype=np.float16).astype(np.float32)

        mat = np.eye(self.num_tickers, dtype=np.float32)
        mat[np.triu_indices(self.num_tickers, k=1)] = triu_vals
        return mat + mat.T - np.eye(self.num_tickers, dtype=np.float32)

    def get_pair_history(self, i: int, j: int) -> np.ndarray:
        """Rolling‑window correlation series for the ticker pair *(i, j)* (float32)."""
        if i == j:
            return np.ones(self.num_corr_days, dtype=np.float32)
        if i > j:
            i, j = j, i

        idx_in_triu = (i * (2 * self.num_tickers - i - 1)) // 2 + (j - i - 1)
        offsets = (
            self._data_offset
            + idx_in_triu * self._ITEM_BYTES
            + np.arange(self.num_corr_days) * self.num_pairs * self._ITEM_BYTES
        )
        out = np.empty(self.num_corr_days, dtype=np.float32)
        for k, pos in enumerate(offsets):
            raw = self._mmap[pos : pos + self._ITEM_BYTES]
            out[k] = np.frombuffer(raw, dtype=np.float16)[0].astype(np.float32)
        return out

    # ───────────────────#
    # helpers functions  #
    # ───────────────────#

    def _write_window(self) -> None:
        """Compute correlations for the *current* window and append to file."""
        corr_vals = self._compute_corr(
            self._sum_cross,
            self._sum_r,
            self._sum_r2,
            self._i_idx,
            self._j_idx,
            self.window_size,
            self.num_pairs,
        ).astype(np.float16)

        win_idx = self.days_written - self.window_size
        start   = self._data_offset + win_idx * self.num_pairs * self._ITEM_BYTES
        self._mmap[start : start + corr_vals.nbytes] = corr_vals.tobytes()

        if win_idx % 100 == 0:  # update header every ~100 windows in case crashes
            self._mmap.seek(8)
            self._mmap.write(np.int32(win_idx + 1).tobytes())
            self._mmap.flush()

    @staticmethod
    def _build_index_arrays(n: int):
        i_idx = np.empty(n * (n - 1) // 2, dtype=np.int32)
        j_idx = np.empty_like(i_idx)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                i_idx[k] = i
                j_idx[k] = j
                k += 1
        return i_idx, j_idx
