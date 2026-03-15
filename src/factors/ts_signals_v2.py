import pandas as pd
import numpy as np


def calc_adx(ohlcv: pd.DataFrame, window: int = 14) -> pd.Series:
    """ADX 趋势强度指标"""
    high = ohlcv['high']
    low = ohlcv['low']
    close = ohlcv['close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0
    mask2 = minus_dm < plus_dm
    minus_dm[mask2] = 0
    
    close_prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(span=window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=window, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.ewm(span=window, adjust=False).mean() / (atr + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=window, adjust=False).mean()
    return adx


def signal_combined_adx(
    ohlcv: pd.DataFrame,
    adx_threshold: float = 20.0,
    require_all: bool = True,
) -> pd.Series:
    """
    组合信号 + ADX 趋势过滤
    只在趋势明确时（ADX > threshold）开仓
    """
    from factors.ts_signals import signal_macd, signal_ma_trend, signal_ma_cross
    
    s1 = signal_macd(ohlcv)
    s2 = signal_ma_trend(ohlcv)
    s3 = signal_ma_cross(ohlcv, fast=20, slow=60)
    adx = calc_adx(ohlcv)
    adx_filter = (adx > adx_threshold).astype(int)
    
    combined = s1 + s2 + s3
    if require_all:
        base = (combined >= 3).astype(int)
    else:
        base = (combined >= 2).astype(int)
    
    # 加 ADX 过滤
    result = (base * adx_filter.shift(1).fillna(0))
    return result.shift(1).fillna(0).astype(int).rename(f'combined_adx_{adx_threshold}')


def signal_ma_cross_adx(
    ohlcv: pd.DataFrame,
    fast: int = 20,
    slow: int = 60,
    adx_threshold: float = 20.0,
) -> pd.Series:
    """均线金叉 + ADX 过滤"""
    close = ohlcv['close']
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    position = (ma_fast > ma_slow).astype(int)
    adx = calc_adx(ohlcv)
    adx_filter = (adx > adx_threshold).astype(int).shift(1).fillna(0)
    result = position * adx_filter
    return result.shift(1).fillna(0).astype(int).rename(f'ma_cross_adx_{fast}_{slow}')


def signal_triple_ma(
    ohlcv: pd.DataFrame,
    fast: int = 10,
    mid: int = 30,
    slow: int = 100,
) -> pd.Series:
    """三线多头排列：fast>mid>slow 且收盘>fast"""
    close = ohlcv['close']
    ma_f = close.rolling(fast).mean()
    ma_m = close.rolling(mid).mean()
    ma_s = close.rolling(slow).mean()
    bull = ((ma_f > ma_m) & (ma_m > ma_s) & (close > ma_f)).astype(int)
    return bull.shift(1).fillna(0).astype(int).rename(f'triple_ma_{fast}_{mid}_{slow}')


def signal_macd_ma_filter(
    ohlcv: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
    ma_filter: int = 200,
) -> pd.Series:
    """MACD + 长期均线过滤（只在MA200上方做多）"""
    close = ohlcv['close']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    hist = macd_line - sig_line
    ma = close.rolling(ma_filter).mean()
    
    macd_signal = (hist > 0).astype(int)
    trend_filter = (close > ma).astype(int).shift(1).fillna(0)
    result = macd_signal * trend_filter
    return result.shift(1).fillna(0).astype(int).rename(f'macd_ma{ma_filter}')
