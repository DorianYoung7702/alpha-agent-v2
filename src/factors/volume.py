"""
Volume Factors - 量价类因子（修复版）
"""
import pandas as pd
import numpy as np
from .base import BaseFactor
from .momentum import _apply_by_symbol


class VolumePriceDivergence(BaseFactor):
    """
    量价相关性：价格变化与成交量变化的滚动相关
    正值=量价同向，负值=量价背离
    """
    def __init__(self, window: int = 10):
        super().__init__(
            name=f"vpd_{window}d",
            description=f"{window}日量价相关性"
        )
        self.window = window

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        def calc(grp):
            price_chg = grp["close"].pct_change()
            vol_chg = grp["volume"].pct_change()
            return price_chg.rolling(self.window).corr(vol_chg)

        results = []
        for sym, grp in panel.groupby(level="symbol"):
            res = calc(grp)
            results.append(res)
        return pd.concat(results).sort_index()


class OBV(BaseFactor):
    """
    OBV 能量潮：N 日变化率
    """
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"obv_{window}d",
            description=f"OBV {window}日变化率"
        )
        self.window = window

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        def calc(grp):
            direction = np.sign(grp["close"].diff())
            obv = (direction * grp["volume"]).cumsum()
            return obv.pct_change(self.window)

        results = []
        for sym, grp in panel.groupby(level="symbol"):
            res = calc(grp)
            results.append(res)
        return pd.concat(results).sort_index()


class MoneyFlow(BaseFactor):
    """
    主动买入占比：taker_buy_quote / quote_volume 的 N 日均值
    """
    def __init__(self, window: int = 5):
        super().__init__(
            name=f"money_flow_{window}d",
            description=f"{window}日主动买入占比"
        )
        self.window = window

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        buy_ratio = panel["taker_buy_quote"] / (panel["quote_volume"] + 1e-10)
        return _apply_by_symbol(
            buy_ratio,
            lambda s: s.rolling(self.window).mean()
        )


class VolumeRatio(BaseFactor):
    """
    量比：当日成交量 / N 日均量
    """
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"vol_ratio_{window}d",
            description=f"成交量/{window}日均量"
        )
        self.window = window

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        def calc(vol):
            avg = vol.rolling(self.window).mean()
            return vol / (avg + 1e-10)

        return _apply_by_symbol(panel["volume"], calc)


class VWAP_Deviation(BaseFactor):
    """
    VWAP 偏离度：当前价格偏离滚动 VWAP 的程度
    """
    def __init__(self, window: int = 10):
        super().__init__(
            name=f"vwap_dev_{window}d",
            description=f"{window}日VWAP偏离度"
        )
        self.window = window

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        def calc(grp):
            vwap = grp["quote_volume"] / (grp["volume"] + 1e-10)
            rolling_vwap = vwap.rolling(self.window).mean()
            return (grp["close"] - rolling_vwap) / (rolling_vwap + 1e-10)

        results = []
        for sym, grp in panel.groupby(level="symbol"):
            res = calc(grp)
            results.append(res)
        return pd.concat(results).sort_index()


class TradeIntensity(BaseFactor):
    """
    交易密度变���率：成交笔数 N 日变化率
    """
    def __init__(self, window: int = 10):
        super().__init__(
            name=f"trade_intensity_{window}d",
            description=f"{window}日成交笔数变化率"
        )
        self.window = window

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        return _apply_by_symbol(
            panel["trades"],
            lambda s: s.pct_change(self.window)
        )
