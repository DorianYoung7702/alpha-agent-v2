"""
TimeSeries Backtest Engine - 时序趋势跟踪回测引擎
单币择时：基于信号做多/空仓，ATR���态止损
"""
import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional

COMMISSION = 0.001
SLIPPAGE = 0.0005
TOTAL_COST = COMMISSION + SLIPPAGE


class TimeSeriesEngine:
    """
    单币时序回���引擎
    信号：+1=做多，0=空仓（不做空）
    执行：信号产生日收盘后，次日开盘价成交
    ���损：持仓后价格跌破 entry_price - ATR*atr_mult 触发
    """
    def __init__(
        self,
        cost_per_side: float = TOTAL_COST,
        atr_window: int = 14,
        atr_mult: float = 2.0,
        use_atr_stop: bool = True,
    ):
        self.cost_per_side = cost_per_side
        self.atr_window = atr_window
        self.atr_mult = atr_mult
        self.use_atr_stop = use_atr_stop

    def run(
        self,
        signal: pd.Series,      # index=timestamp, values=+1/0，已shift防前视
        ohlcv: pd.DataFrame,    # index=timestamp, cols=open/high/low/close/volume
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if start_date:
            signal = signal[signal.index >= start_date]
            ohlcv = ohlcv[ohlcv.index >= start_date]
        if end_date:
            signal = signal[signal.index <= end_date]
            ohlcv = ohlcv[ohlcv.index <= end_date]

        # 计算 ATR
        atr = self._calc_atr(ohlcv)

        dates = sorted(set(signal.index) & set(ohlcv.index))
        if len(dates) < 20:
            return pd.DataFrame()

        results = []
        position = 0       # 当前仓位：0=空仓，1=持仓
        entry_price = 0.0  # 买入价
        entry_atr = 0.0    # 买入时的ATR

        for i, date in enumerate(dates):
            if i == 0:
                continue

            prev_date = dates[i - 1]
            sig = signal.get(prev_date, 0)  # 昨日信号，今日执行
            open_price = ohlcv.loc[date, "open"]
            close_price = ohlcv.loc[date, "close"]
            low_price = ohlcv.loc[date, "low"]
            curr_atr = atr.get(date, 0)

            daily_ret = 0.0
            cost = 0.0
            stopped = False

            if position == 1:
                # ATR 止损检查（用当日最低价）
                if self.use_atr_stop and entry_atr > 0:
                    stop_level = entry_price - self.atr_mult * entry_atr
                    if low_price <= stop_level:
                        # 止损：用止损价成交
                        exit_price = max(stop_level, low_price)
                        daily_ret = (exit_price - entry_price) / entry_price
                        cost = self.cost_per_side  # 平仓成本
                        position = 0
                        stopped = True
                        logger.debug(f"[Stop] {date} entry={entry_price:.0f} stop={stop_level:.0f} exit={exit_price:.0f}")

                if not stopped:
                    # 正常持仓收益
                    prev_close = ohlcv.loc[prev_date, "close"]
                    daily_ret = (close_price - prev_close) / prev_close

                    # 信号变为0：今日收盘平仓
                    if sig == 0:
                        cost = self.cost_per_side
                        position = 0

            elif position == 0:
                if sig == 1:
                    # 今日开盘买入
                    entry_price = open_price
                    entry_atr = curr_atr
                    # 开盘到收盘收益
                    daily_ret = (close_price - open_price) / open_price
                    cost = self.cost_per_side  # 买入成本
                    position = 1

            net_ret = daily_ret - cost
            results.append({
                "timestamp": date,
                "return": net_ret,
                "gross_return": daily_ret,
                "cost": cost,
                "position": position,
                "stopped": stopped,
                "close": close_price,
            })

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).set_index("timestamp")

    def _calc_atr(self, ohlcv: pd.DataFrame) -> pd.Series:
        high = ohlcv["high"]
        low = ohlcv["low"]
        close_prev = ohlcv["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(self.atr_window, min_periods=self.atr_window // 2).mean()
