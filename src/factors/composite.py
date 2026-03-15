"""
Composite Factor + Market Filter
多因子合成 + 市场状态过滤
"""
import pandas as pd
import numpy as np
from loguru import logger
from typing import List, Tuple
from .base import BaseFactor


class MarketFilter:
    """
    市场状态过滤器
    只在趋势向上时做多，避免熊市裸多头亏损
    """
    def __init__(self, ma_window: int = 200, btc_symbol: str = "BTCUSDT"):
        self.ma_window = ma_window
        self.btc_symbol = btc_symbol

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        计算市场状态：1=做多, 0=空仓
        条件：BTC 收盘价 > N日均线
        强制 shift(1) 防未来函数
        """
        try:
            btc_close = panel["close"].xs(self.btc_symbol, level="symbol")
        except KeyError:
            # 找���一个可用的 symbol
            symbols = panel.index.get_level_values("symbol").unique()
            btc_close = panel["close"].xs(symbols[0], level="symbol")

        ma = btc_close.rolling(self.ma_window).mean()
        signal = (btc_close > ma).astype(float)
        # shift(1): 用昨日状态决定今日是否交易
        return signal.shift(1).rename("market_filter")


class CompositeFactor:
    """
    多因子 IC 加权合成
    用历史 IC 作为权重，动���调整各因子贡献
    """
    def __init__(
        self,
        factors: List[BaseFactor],
        ic_window: int = 60,
        use_equal_weight: bool = False
    ):
        self.factors = factors
        self.ic_window = ic_window
        self.use_equal_weight = use_equal_weight

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        计算合成因子
        1. 计算各因子值（已含 shift+标准化）
        2. 用等权或历史 IC 加权合成
        """
        factor_values = {}
        for f in self.factors:
            try:
                val = f.compute(panel)
                factor_values[f.name] = val
            except Exception as e:
                logger.warning(f"Factor {f.name} failed: {e}")

        if not factor_values:
            raise ValueError("No valid factors computed")

        if self.use_equal_weight or len(factor_values) == 1:
            # 等权合成
            composite = pd.concat(list(factor_values.values()), axis=1).mean(axis=1)
        else:
            # IC 加权合成
            composite = self._ic_weighted(panel, factor_values)

        return composite.rename("composite")

    def _ic_weighted(
        self,
        panel: pd.DataFrame,
        factor_values: dict
    ) -> pd.Series:
        """
        用历史滚动 IC 加权
        权重 = max(0, rolling_IC)，负 IC 因子不参与
        """
        from backtest.metrics import calc_forward_returns, calc_ic_series

        fwd_ret = calc_forward_returns(panel, periods=1)
        weights = {}

        for name, fval in factor_values.items():
            try:
                ic_series = calc_ic_series(fval, fwd_ret)
                # 用滚动均值作为权重估计
                rolling_ic = ic_series.rolling(self.ic_window, min_periods=10).mean()
                weights[name] = rolling_ic
            except Exception:
                weights[name] = pd.Series(0.0, index=fval.index.get_level_values("timestamp").unique())

        # 按时间截面加权合成
        dates = panel.index.get_level_values("timestamp").unique()
        result_frames = []

        for date in dates:
            day_weights = {}
            for name, w_series in weights.items():
                if date in w_series.index:
                    w = float(w_series.loc[date])
                    day_weights[name] = max(0.0, w)  # 负 IC 不参与

            total_w = sum(day_weights.values())
            if total_w < 1e-6:
                # 权重全为 0 时等权
                for name in factor_values:
                    day_weights[name] = 1.0 / len(factor_values)
                total_w = 1.0

            # 加权合成当日因子值
            try:
                day_composite = None
                for name, w in day_weights.items():
                    fval_day = factor_values[name].xs(date, level="timestamp") * (w / total_w)
                    if day_composite is None:
                        day_composite = fval_day
                    else:
                        day_composite = day_composite.add(fval_day, fill_value=0)

                if day_composite is not None:
                    day_composite.index = pd.MultiIndex.from_tuples(
                        [(date, sym) for sym in day_composite.index],
                        names=["timestamp", "symbol"]
                    )
                    result_frames.append(day_composite)
            except Exception:
                continue

        if not result_frames:
            # fallback: 等权
            return pd.concat(list(factor_values.values()), axis=1).mean(axis=1)

        return pd.concat(result_frames).sort_index()
