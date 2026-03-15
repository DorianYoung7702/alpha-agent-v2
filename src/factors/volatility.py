"""
Volatility Factors - 波动率类因子（修复版）
"""
import pandas as pd
import numpy as np
from .base import BaseFactor
from .momentum import _apply_by_symbol


class HistoricalVolatility(BaseFactor):
    """
    历史波动率：N 日对数收益率标准差（年化）
    取负值：低波动率得高分
    """
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"hvol_{window}d",
            description=f"{window}日历史波动率（取负）"
        )
        self.window = window

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        def calc(prices):
            log_ret = np.log(prices / prices.shift(1))
            hvol = log_ret.rolling(self.window, min_periods=self.window//2).std() * np.sqrt(252)
            return -hvol  # 低波动得高分

        return _apply_by_symbol(panel["close"], calc)


class ATR(BaseFactor):
    """
    ATR 归一化：平均真实波动范��� / 收盘价
    """
    def __init__(self, window: int = 14):
        super().__init__(
            name=f"atr_{window}d",
            description=f"{window}日ATR/Close"
        )
        self.window = window

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        results = []
        for sym, grp in panel.groupby(level="symbol"):
            high = grp["high"]
            low = grp["low"]
            close_prev = grp["close"].shift(1)
            tr = pd.concat([
                high - low,
                (high - close_prev).abs(),
                (low - close_prev).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(self.window, min_periods=self.window//2).mean()
            result = atr / (grp["close"] + 1e-10)
            results.append(result)
        return pd.concat(results).sort_index()


class BollingerBandWidth(BaseFactor):
    """
    布林带宽度：(上轨 - 下轨) / 中轨
    """
    def __init__(self, window: int = 20, n_std: float = 2.0):
        super().__init__(
            name=f"bbw_{window}d",
            description=f"布林带宽度 {window}日"
        )
        self.window = window
        self.n_std = n_std

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        def calc(prices):
            mid = prices.rolling(self.window, min_periods=self.window//2).mean()
            std = prices.rolling(self.window, min_periods=self.window//2).std()
            upper = mid + self.n_std * std
            lower = mid - self.n_std * std
            return (upper - lower) / (mid + 1e-10)

        return _apply_by_symbol(panel["close"], calc)


class VolatilityRatio(BaseFactor):
    """
    波动率比率：短期波动 / 长期波动
    """
    def __init__(self, short: int = 5, long: int = 20):
        super().__init__(
            name=f"volratio_{short}_{long}",
            description=f"波动率比率({short}日/{long}日)"
        )
        self.short = short
        self.long = long

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        def calc(prices):
            log_ret = np.log(prices / prices.shift(1))
            vol_s = log_ret.rolling(self.short, min_periods=self.short//2).std()
            vol_l = log_ret.rolling(self.long, min_periods=self.long//2).std()
            return vol_s / (vol_l + 1e-10)

        return _apply_by_symbol(panel["close"], calc)


class DownsideVolatility(BaseFactor):
    """
    下行波动率：只计算负收益的标准差，取负值
    """
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"downvol_{window}d",
            description=f"{window}日下行波动率（取负）"
        )
        self.window = window

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        def calc(prices):
            ret = prices.pct_change()
            downside = ret.where(ret < 0, 0.0)
            dvol = downside.rolling(self.window, min_periods=self.window//2).std() * np.sqrt(252)
            return -dvol  # 低下行波动得高分

        return _apply_by_symbol(panel["close"], calc)
