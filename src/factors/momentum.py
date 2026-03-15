"""
Momentum Factors - 动量类因子（修复版）
所有因子严格使用逐 symbol 循环，确保无未���函数
"""
import pandas as pd
import numpy as np
from .base import BaseFactor


def _apply_by_symbol(panel_or_series, func):
    """
    安全的逐 symbol 计算辅助函数
    避免 groupby.apply 的 index 重组问题
    """
    results = []
    grouped = panel_or_series.groupby(level="symbol")
    for sym, grp in grouped:
        res = func(grp)
        results.append(res)
    return pd.concat(results).sort_index()


class ReturnMomentum(BaseFactor):
    """
    N 日价格动量：过去 N 日收益率
    基类 shift(1) 后 = 用 t-1 日数据，t 日执行
    """
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"mom_{window}d",
            description=f"{window}日价格动量"
        )
        self.window = window

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        return _apply_by_symbol(
            panel["close"],
            lambda s: s.pct_change(self.window)
        )


class RSIMomentum(BaseFactor):
    """
    RSI：相对强弱指数（Wilder 平滑法，无未来函数）
    """
    def __init__(self, window: int = 14):
        super().__init__(
            name=f"rsi_{window}",
            description=f"{window}日RSI"
        )
        self.window = window

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        def calc_rsi(prices: pd.Series) -> pd.Series:
            delta = prices.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(com=self.window - 1, min_periods=self.window).mean()
            avg_loss = loss.ewm(com=self.window - 1, min_periods=self.window).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            return 100 - (100 / (1 + rs))
        return _apply_by_symbol(panel["close"], calc_rsi)


class MACDSignal(BaseFactor):
    """
    MACD 柱状图 = MACD线 - 信号线（全 EMA，无未来函数）
    """
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(
            name=f"macd_{fast}_{slow}_{signal}",
            description=f"MACD柱状图({fast},{slow},{signal})"
        )
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        def calc_macd(prices: pd.Series) -> pd.Series:
            ema_fast = prices.ewm(span=self.fast, adjust=False).mean()
            ema_slow = prices.ewm(span=self.slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
            return macd_line - signal_line
        return _apply_by_symbol(panel["close"], calc_macd)


class ShortTermReversal(BaseFactor):
    """1日短期反转"""
    def __init__(self):
        super().__init__(name="reversal_1d", description="1日短期反转")

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        return _apply_by_symbol(panel["close"], lambda s: -s.pct_change(1))


class PriceAcceleration(BaseFactor):
    """价格加速度：短期动量 - 长期动量"""
    def __init__(self, short: int = 5, long: int = 20):
        super().__init__(
            name=f"accel_{short}_{long}",
            description=f"价格���速度({short}日-{long}日)"
        )
        self.short = short
        self.long = long

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        def calc(prices):
            return prices.pct_change(self.short) - prices.pct_change(self.long)
        return _apply_by_symbol(panel["close"], calc)


class FundingRateMomentum(BaseFactor):
    """���金费率顺势因子：费率持续为正 → 多头情绪强 → 顺势做多"""
    def __init__(self, window: int = 7):
        super().__init__(
            name=f"funding_mom_{window}d",
            description=f"{window}日资金费率���势"
        )
        self.window = window

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        if "funding_rate" not in panel.columns:
            raise ValueError("Panel missing 'funding_rate' column")
        return _apply_by_symbol(
            panel["funding_rate"],
            lambda s: s.rolling(self.window, min_periods=self.window // 2).mean()
        )


class FundingRateReversal(BaseFactor):
    """
    资金费率均值回归因子（加密市场独��� alpha）
    逻辑：高正费率 = 多头拥挤 = 现货被高估，即将回调，做空
          高负费率 = 空头拥挤 = 现货被低估，即将反弹，做多
    公式：FR_score = -1 × MA(FundingRate_daily, window)
    数据来源���币安永续合约 fundingRate（已截断 ±0.5%）
    预期 IC：0.05~0.10（震荡市最强）
    """
    def __init__(self, window: int = 7):
        super().__init__(
            name=f"fr_reversal_{window}d",
            description=f"资金费率{window}日均值反转"
        )
        self.window = window

    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        if "funding_rate" not in panel.columns:
            raise ValueError("Panel missing 'funding_rate' column")
        def calc(fr):
            # 取负值：高正费率得低分（看空），高负费率得高分（看���）
            return -fr.rolling(self.window, min_periods=self.window // 2).mean()
        return _apply_by_symbol(panel["funding_rate"], calc)
