"""
M1: Data Loader & Quantizer
Reads parquet files from is_features_1_min_hackaton / if_features_5_min_hackaton
and converts wide-format (timestamp x tickers) to long-format bars.
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Tuple, Dict


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SESSION_START = "10:00"
SESSION_END   = "18:30"
TZ            = "Europe/Moscow"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _load_parquet_folder(folder: str) -> Dict[str, pd.DataFrame]:
    """Load all .parquet files from a folder → {filename_stem: DataFrame}
    Strips trailing underscores from stems so 'open_.parquet' → key 'open'.
    """
    result = {}
    pattern = os.path.join(folder, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in: {folder}")
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0].rstrip("_")
        df = pd.read_parquet(fp)
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(TZ)
        else:
            df.index = df.index.tz_convert(TZ)
        df.index.name = "timestamp"
        result[stem] = df
        print(f"  Loaded '{stem}': {df.shape[0]} rows × {df.shape[1]} cols")
    return result


def _wide_to_long(wide_df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Convert wide DataFrame (timestamp × tickers) → long DataFrame
    with columns [timestamp, seccode, <value_name>].
    """
    long = (
        wide_df
        .stack(future_stack=True)      # (timestamp, seccode) → value
        .reset_index()
    )
    long.columns = ["timestamp", "seccode", value_name]
    return long


def _session_filter(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Keep only rows inside 10:00–18:30 MSK."""
    t = df[ts_col]
    time_of_day = t.dt.hour * 60 + t.dt.minute
    start = 10 * 60
    end   = 18 * 60 + 30
    return df[(time_of_day >= start) & (time_of_day <= end)].copy()


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
class DataLoader:
    """
    Loads and prepares 1-min and 5-min bar data from parquet folders.

    Expected folder layout (each file named after what it stores):
        if_features_5_min_hackaton/
            open.parquet        (timestamp × tickers)
            high.parquet
            low.parquet
            close.parquet
            volume.parquet
            ... (any extra features)

        is_features_1_min_hackaton/
            open.parquet
            high.parquet
            low.parquet
            close.parquet
            volume.parquet
            ... (spread, bid, ask, etc.)
    """

    def __init__(self, folder_1min: str, folder_5min: str):
        self.folder_1min = folder_1min
        self.folder_5min = folder_5min
        self._raw_1min: Dict[str, pd.DataFrame] = {}
        self._raw_5min: Dict[str, pd.DataFrame] = {}

    def load(self) -> "DataLoader":
        print("Loading 5-min data …")
        self._raw_5min = _load_parquet_folder(self.folder_5min)
        print("Loading 1-min data …")
        self._raw_1min = _load_parquet_folder(self.folder_1min)
        return self

    # ── 5-min bars ──────────────────────────────────────────────────────────
    def bars_5min(self) -> pd.DataFrame:
        """
        Returns long-format 5-min OHLCV DataFrame:
        columns: [timestamp, seccode, open, high, low, close, volume, mid, ...]
        """
        dfs = {}
        for key in ("open", "high", "low", "close", "volume"):
            if key not in self._raw_5min:
                if key == "volume":
                    # volume может отсутствовать в 5-мин данных — подставляем нули
                    first = next(iter(self._raw_5min.values()))
                    dummy = pd.DataFrame(0, index=first.index, columns=first.columns)
                    dfs[key] = _wide_to_long(dummy, key)
                else:
                    raise KeyError(f"Missing '{key}.parquet' in 5-min folder")
            else:
                dfs[key] = _wide_to_long(self._raw_5min[key], key)

        merged = dfs["open"]
        for key in ("high", "low", "close", "volume"):
            merged = merged.merge(dfs[key], on=["timestamp", "seccode"], how="outer")

        merged = _session_filter(merged)
        merged["mid"] = (merged["open"] + merged["close"]) / 2
        merged["bar_end_ts"] = (merged["timestamp"] + pd.Timedelta(minutes=5)).astype("int64")
        merged = merged.sort_values(["timestamp", "seccode"]).reset_index(drop=True)

        # Attach any extra features (e.g. spread, imbalance)
        extra_keys = [k for k in self._raw_5min if k not in ("open","high","low","close","volume")]
        for k in extra_keys:
            tmp = _wide_to_long(self._raw_5min[k], k)
            merged = merged.merge(tmp, on=["timestamp", "seccode"], how="left")

        return merged

    # ── 1-min bars ──────────────────────────────────────────────────────────
    def bars_1min(self) -> pd.DataFrame:
        """
        Returns long-format 1-min OHLCV DataFrame.
        """
        dfs = {}
        for key in ("open", "high", "low", "close", "volume"):
            if key not in self._raw_1min:
                if key == "volume":
                    # volume может отсутствовать в 5-мин данных — подставляем нули
                    first = next(iter(self._raw_1min.values()))
                    dummy = pd.DataFrame(0, index=first.index, columns=first.columns)
                    dfs[key] = _wide_to_long(dummy, key)
                else:
                    raise KeyError(f"Missing '{key}.parquet' in 5-min folder")
            else:
                dfs[key] = _wide_to_long(self._raw_1min[key], key)

        merged = dfs["open"]
        for key in ("high", "low", "close", "volume"):
            merged = merged.merge(dfs[key], on=["timestamp", "seccode"], how="outer")

        merged = _session_filter(merged)
        merged["mid"] = (merged["open"] + merged["close"]) / 2
        merged["bar_end_ts"] = (merged["timestamp"] + pd.Timedelta(minutes=1)).astype("int64")
        merged = merged.sort_values(["timestamp", "seccode"]).reset_index(drop=True)

        # Extra 1-min features (bid/ask spread etc.)
        extra_keys = [k for k in self._raw_1min if k not in ("open","high","low","close","volume")]
        for k in extra_keys:
            tmp = _wide_to_long(self._raw_1min[k], k)
            merged = merged.merge(tmp, on=["timestamp", "seccode"], how="left")

        return merged

    # ── Ticker universe ──────────────────────────────────────────────────────
    def tickers(self) -> list:
        if not self._raw_5min:
            raise RuntimeError("Call .load() first")
        first = next(iter(self._raw_5min.values()))
        return sorted(first.columns.tolist())

    # ── ADV (Average Daily Volume) ───────────────────────────────────────────
    def adv(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ADV (average daily volume) per ticker from bars_5min or bars_1min.
        Returns DataFrame: [seccode, adv]
        """
        daily = (
            bars.groupby(["seccode", bars["timestamp"].dt.date])["volume"]
            .sum()
            .reset_index()
        )
        return daily.groupby("seccode")["volume"].mean().rename("adv").reset_index()
