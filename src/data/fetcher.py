"""
Data Fetcher v2 - 动态币池支持
每个截面日取当日市值排名前 N 的币，要求有足够历���数据
"""
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Optional, List

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BINANCE_BASE = "https://api.binance.com"
BINANCE_FAPI = "https://fapi.binance.com"

# 扩展到 Top30 市值币（稳定的，有合约的）
SYMBOLS_TOP30 = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT",
    "ATOMUSDT", "UNIUSDT", "ETCUSDT", "XLMUSDT", "NEARUSDT",
    "ALGOUSDT", "FTMUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT",
    "GALAUSDT", "APEUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    "INJUSDT", "SUIUSDT", "SEIUSDT", "TIAUSDT", "WLDUSDT",
]

# 原始 Top10（向后兼容）
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT"
]


def fetch_klines(
    symbol: str,
    interval: str = "1d",
    start_str: str = "2020-01-01",
    end_str: Optional[str] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    cache_path = RAW_DIR / f"{symbol}_{interval}_{start_str[:10]}.parquet"
    if use_cache and cache_path.exists():
        logger.debug(f"[Cache] {symbol} {interval}")
        return pd.read_parquet(cache_path)

    logger.info(f"[Fetch] {symbol} {interval} from {start_str}")
    url = f"{BINANCE_BASE}/api/v3/klines"
    all_data = []
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_str).timestamp() * 1000) if end_str else int(time.time() * 1000)

    while start_ts < end_ts:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": start_ts, "endTime": end_ts, "limit": 1000}
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            start_ts = data[-1][0] + 1
            if len(data) < 1000:
                break
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error {symbol}: {e}")
            time.sleep(5)
            continue

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df = df[numeric_cols]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.to_parquet(cache_path)
    logger.info(f"[Saved] {symbol} {len(df)} rows")
    return df


def fetch_funding_rate(
    symbol: str,
    start_str: str = "2020-01-01",
    use_cache: bool = True
) -> pd.DataFrame:
    """
    获取资金费率（永续合约，8小时一次，聚合为日均，并截断异常值）
    """
    cache_path = RAW_DIR / f"{symbol}_funding_{start_str[:10]}.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    logger.info(f"[Fetch] {symbol} funding rate")
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
    all_data = []
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(time.time() * 1000)

    while start_ts < end_ts:
        params = {"symbol": symbol, "startTime": start_ts, "limit": 1000}
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data or isinstance(data, dict):
                break
            all_data.extend(data)
            start_ts = data[-1]["fundingTime"] + 1
            if len(data) < 1000:
                break
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Funding rate error {symbol}: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms").dt.normalize()
    df["fundingRate"] = df["fundingRate"].astype(float)

    # 截断异常值：���次费率 clip 到 ±0.5%
    df["fundingRate"] = df["fundingRate"].clip(-0.005, 0.005)

    # 聚合为日均（一天3次 × 日均）
    daily = df.groupby("timestamp")["fundingRate"].mean().rename("funding_rate")
    result = daily.to_frame()
    result.to_parquet(cache_path)
    logger.info(f"[Saved] {symbol} funding {len(result)} days")
    return result


def fetch_all_symbols(
    interval: str = "1d",
    start_str: str = "2020-01-01",
    symbols: list = None,
    use_cache: bool = True
) -> dict:
    if symbols is None:
        symbols = SYMBOLS_TOP30
    result = {}
    for sym in symbols:
        df = fetch_klines(sym, interval, start_str, use_cache=use_cache)
        if not df.empty:
            result[sym] = df
        time.sleep(0.15)
    logger.info(f"[Done] Fetched {len(result)}/{len(symbols)} symbols")
    return result


def fetch_funding_all(
    start_str: str = "2020-01-01",
    symbols: list = None,
    use_cache: bool = True,
    min_volume_usdt: float = 5e7  # 流动性过滤：日均合约成交量 > 5000万
) -> dict:
    """
    ���量获取资金费率，带流动性过滤
    """
    if symbols is None:
        symbols = SYMBOLS_TOP30
    result = {}
    for sym in symbols:
        df = fetch_funding_rate(sym, start_str, use_cache=use_cache)
        if not df.empty:
            result[sym] = df
        time.sleep(0.15)
    logger.info(f"[Funding] Fetched {len(result)}/{len(symbols)} symbols")
    return result


def build_panel(
    data: dict,
    funding_data: dict = None,
    min_history_days: int = 60
) -> pd.DataFrame:
    """
    构建面板数据
    min_history_days: 某币在某日必须有至少N天历史才进���截面
    """
    frames = []
    for sym, df in data.items():
        df = df.copy()
        df["symbol"] = sym
        # 加入资金费率
        if funding_data and sym in funding_data:
            fr = funding_data[sym]["funding_rate"]
            df = df.join(fr, how="left")
        frames.append(df)

    panel = pd.concat(frames)
    panel.index.name = "timestamp"
    panel = panel.reset_index().set_index(["timestamp", "symbol"]).sort_index()

    # 应用 min_history_days 过滤
    # 某币在某日的行，只有当该币在该日之前有 >= min_history_days 天数据才保留
    panel = _apply_min_history_filter(panel, min_history_days)

    logger.info(f"[Panel] Shape after filter: {panel.shape}")
    return panel


def _apply_min_history_filter(panel: pd.DataFrame, min_days: int) -> pd.DataFrame:
    """
    ���滤掉历史数据不足的（symbol, date）行
    """
    # 计算每个 symbol 的数据起始日期
    sym_start = {}
    for sym in panel.index.get_level_values("symbol").unique():
        try:
            sym_data = panel.xs(sym, level="symbol")
            sym_start[sym] = sym_data.index.min()
        except Exception:
            continue

    # 过滤：date - sym_start >= min_days
    keep_mask = []
    for (date, sym) in panel.index:
        start = sym_start.get(sym, date)
        days_available = (date - start).days
        keep_mask.append(days_available >= min_days)

    return panel[keep_mask]
