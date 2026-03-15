"""
TimeSeries Signals - 时序择时信号
所有信号强制 shift(1)，防未来函数
返回：pd.Series���index=timestamp，values=+1(做多)/0(空仓)
"""
import pandas as pd
import numpy as np


def signal_macd(
    ohlcv: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """
    MACD 金���做多，死叉空仓
    金叉：MACD线从下穿越信号线
    死叉：MACD线从上穿越信号线
    """
    close = ohlcv["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    # 金叉：hist由负转正
    position = (hist > 0).astype(int)
    # shift(1)：今日���号明日执行
    return position.shift(1).fillna(0).astype(int).rename("macd_signal")


def signal_bollinger(
    ohlcv: pd.DataFrame,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.Series:
    """
    布林带突破策略
    做多：收盘价突破上轨
    空仓：收盘价跌破中轨
    """
    close = ohlcv["close"]
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + n_std * std

    position = pd.Series(0, index=close.index)
    in_long = False
    for i in range(len(close)):
        if pd.isna(upper.iloc[i]):
            position.iloc[i] = 0
            continue
        if not in_long and close.iloc[i] > upper.iloc[i]:
            in_long = True
        elif in_long and close.iloc[i] < mid.iloc[i]:
            in_long = False
        position.iloc[i] = 1 if in_long else 0

    return position.shift(1).fillna(0).astype(int).rename("bb_signal")


def signal_ma_trend(
    ohlcv: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    trend: int = 200,
) -> pd.Series:
    """
    均线多头排列：MA20 > MA50 > MA200 且收盘 > MA20
    任一条件不满足则空仓
    """
    close = ohlcv["close"]
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    ma_trend = close.rolling(trend).mean()

    bull = (
        (ma_fast > ma_slow) &
        (ma_slow > ma_trend) &
        (close > ma_fast)
    ).astype(int)

    return bull.shift(1).fillna(0).astype(int).rename("ma_trend_signal")


def signal_ma_cross(
    ohlcv: pd.DataFrame,
    fast: int = 20,
    slow: int = 60,
) -> pd.Series:
    """
    均线金叉死叉
    金叉（MA快线上穿慢线）：做多
    死叉（MA快线下穿慢线）：空仓
    """
    close = ohlcv["close"]
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()

    position = (ma_fast > ma_slow).astype(int)
    return position.shift(1).fillna(0).astype(int).rename(f"ma_cross_{fast}_{slow}")


def signal_rsi_trend(
    ohlcv: pd.DataFrame,
    rsi_window: int = 14,
    oversold: float = 40,
    overbought: float = 70,
) -> pd.Series:
    """
    RSI 趋势过滤：RSI > oversold 且上升趋势时做多
    RSI < overbought 时平���（过热信号）
    """
    close = ohlcv["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=rsi_window-1, min_periods=rsi_window).mean()
    avg_loss = loss.ewm(com=rsi_window-1, min_periods=rsi_window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    position = ((rsi > oversold) & (rsi < overbought)).astype(int)
    return position.shift(1).fillna(0).astype(int).rename(f"rsi_trend_{rsi_window}")


def signal_combined(
    ohlcv: pd.DataFrame,
    require_all: bool = False,
) -> pd.Series:
    """
    组合信号：MACD + MA趋势 同时确认
    require_all=True：所有信号都为1才做多
    require_all=False：多数信号为1就做多
    """
    s1 = signal_macd(ohlcv)
    s2 = signal_ma_trend(ohlcv)
    s3 = signal_ma_cross(ohlcv, fast=20, slow=60)

    combined = s1 + s2 + s3
    if require_all:
        return (combined >= 3).astype(int).rename("combined_all")
    else:
        return (combined >= 2).astype(int).rename("combined_majority")
