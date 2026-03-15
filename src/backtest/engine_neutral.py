"""
Market Neutral Engine - 市场中性回测引擎
做多 Top N + 做空 Bottom N，天然对冲市场风险
最大回撤主要来自因子 Alpha，而非市场 Beta
需要合约账户支持做空
"""
import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional

COMMISSION = 0.001
SLIPPAGE = 0.0005
TOTAL_COST = COMMISSION + SLIPPAGE


class MarketNeutralEngine:
    """
    市场中性回测引擎
    每日：
      - 做多因子���分最高的 Top N 个币（等权）
      - 做空因子得分最低的 Bottom N 个币（等权）
      - 多空各占 50% 资金，总敞口 100%
    """
    def __init__(
        self,
        top_n: int = 3,
        cost_per_side: float = TOTAL_COST,
        min_symbols: int = 8,  # 最少需要多少个币才能建仓
    ):
        self.top_n = top_n
        self.cost_per_side = cost_per_side
        self.min_symbols = min_symbols

    def run(
        self,
        factor: pd.Series,
        panel: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if start_date:
            factor = factor[factor.index.get_level_values("timestamp") >= start_date]
            panel = panel[panel.index.get_level_values("timestamp") >= start_date]
        if end_date:
            factor = factor[factor.index.get_level_values("timestamp") <= end_date]
            panel = panel[panel.index.get_level_values("timestamp") <= end_date]

        dates = sorted(factor.index.get_level_values("timestamp").unique())
        if len(dates) < 20:
            return pd.DataFrame()

        prices = panel["close"].unstack(level="symbol")
        daily_returns = []
        prev_long = {}
        prev_short = {}

        for i, date in enumerate(dates):
            if i == 0:
                prev_long, prev_short = self._select_positions(factor, date)
                continue

            curr_long, curr_short = self._select_positions(factor, date)
            prev_date = dates[i - 1]

            if prev_date not in prices.index or date not in prices.index:
                continue

            price_today = prices.loc[date]
            price_prev = prices.loc[prev_date]
            pct_chg = (price_today - price_prev) / (price_prev + 1e-10)

            # 多头收益
            long_ret = 0.0
            for sym, w in prev_long.items():
                if sym in pct_chg.index and not np.isnan(pct_chg[sym]):
                    long_ret += w * pct_chg[sym]

            # 空头收益（做空：价格下跌赚钱）
            short_ret = 0.0
            for sym, w in prev_short.items():
                if sym in pct_chg.index and not np.isnan(pct_chg[sym]):
                    short_ret += w * (-pct_chg[sym])  # 做空收益取���

            # 换手成本
            long_turnover = self._calc_turnover(
                pd.Series(prev_long), pd.Series(curr_long)
            )
            short_turnover = self._calc_turnover(
                pd.Series(prev_short), pd.Series(curr_short)
            )
            total_cost = (long_turnover + short_turnover) * self.cost_per_side * 2

            # 组合净收益 = 多头50% + 空头50% - 成本
            portfolio_ret = (long_ret * 0.5 + short_ret * 0.5) - total_cost

            daily_returns.append({
                "timestamp": date,
                "return": portfolio_ret,
                "long_ret": long_ret,
                "short_ret": short_ret,
                "cost": total_cost,
                "long_syms": list(curr_long.keys()),
                "short_syms": list(curr_short.keys()),
            })

            prev_long = curr_long
            prev_short = curr_short

        if not daily_returns:
            return pd.DataFrame()

        return pd.DataFrame(daily_returns).set_index("timestamp")

    def _select_positions(self, factor: pd.Series, date) -> tuple:
        """选出当日多头和空头持仓"""
        try:
            day_factor = factor.xs(date, level="timestamp").dropna()
        except KeyError:
            return {}, {}

        if len(day_factor) < self.min_symbols:
            return {}, {}

        sorted_f = day_factor.sort_values(ascending=False)
        top = sorted_f.iloc[:self.top_n]
        bottom = sorted_f.iloc[-self.top_n:]

        # 确保多空不重叠
        if set(top.index) & set(bottom.index):
            return {}, {}

        long_w = {sym: 1.0 / self.top_n for sym in top.index}
        short_w = {sym: 1.0 / self.top_n for sym in bottom.index}
        return long_w, short_w

    def _calc_turnover(self, prev: pd.Series, curr: pd.Series) -> float:
        all_syms = set(prev.index) | set(curr.index)
        turnover = sum(abs(curr.get(s, 0.0) - prev.get(s, 0.0)) for s in all_syms)
        return turnover / 2
