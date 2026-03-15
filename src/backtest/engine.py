"""
Backtest Engine v2 - 单币止损版
截面因子选币策略，防未来函数，含单币止损（不用���合级止损）
expert 建议：组合止损导致全程空仓，改为单币 -15% 止损只砍单币
"""
import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional

COMMISSION = 0.001
SLIPPAGE = 0.0005
TOTAL_COST = COMMISSION + SLIPPAGE


class BacktestEngine:
    """
    截面因子回测引擎（单币止损版）
    ���辑：
      1. 每日用 t-1 日因子（已 shift）对币池排序
      2. 做多 Top N 个币，等权重
      3. 用 t 日收盘价计算收益（t+1 日开盘执行，近似用收盘）
      4. 单币持仓后跌幅超过 stop_loss_single 则次日清仓，不影响其他仓位
      5. 扣除双边手续费 + 滑点
    """
    def __init__(
        self,
        top_n: int = 8,           # Top3→Top8，分散化降噪
        exec_price: str = "open",   # 必须用open：信号t日收盘计算，t+1日开盘执行
        cost_per_side: float = TOTAL_COST,
        stop_loss_single: float = -0.15,  # 单币止损：持仓后跌 -15% 清仓
        rebalance_every_n_days: int = 3,  # 3日换仓
    ):
        self.top_n = top_n
        self.exec_price = exec_price
        self.cost_per_side = cost_per_side
        self.stop_loss_single = stop_loss_single
        self.rebalance_every_n_days = rebalance_every_n_days

    def run(
        self,
        factor: pd.Series,
        panel: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        market_filter: pd.Series = None
    ) -> pd.DataFrame:
        if start_date:
            factor = factor[factor.index.get_level_values("timestamp") >= start_date]
            panel = panel[panel.index.get_level_values("timestamp") >= start_date]
        if end_date:
            factor = factor[factor.index.get_level_values("timestamp") <= end_date]
            panel = panel[panel.index.get_level_values("timestamp") <= end_date]

        dates = sorted(factor.index.get_level_values("timestamp").unique())
        if len(dates) < 20:
            logger.warning("Too few dates for backtest")
            return pd.DataFrame()

        prices = panel[self.exec_price].unstack(level="symbol")
        holdings = self._compute_holdings(factor, dates)
        if not isinstance(holdings, pd.DataFrame) or holdings.empty:
            return pd.DataFrame()

        return self._compute_returns(holdings, prices, dates, market_filter)

    def _compute_holdings(self, factor: pd.Series, dates: list) -> pd.DataFrame:
        records = []
        last_rebalance_idx = -999  # 强制第一天建仓
        last_weights = {}
        for i, date in enumerate(dates):
            # 3日换仓：只在换仓日重新计算持仓
            if (i - last_rebalance_idx) < self.rebalance_every_n_days and last_weights:
                weights = dict(last_weights)
                weights["timestamp"] = date
                records.append(weights)
                continue
            try:
                day_factor = factor.xs(date, level="timestamp").dropna()
                if len(day_factor) < self.top_n:
                    if last_weights:
                        weights = dict(last_weights)
                        weights["timestamp"] = date
                        records.append(weights)
                    continue
                top = day_factor.nlargest(self.top_n)
                weights = {sym: 1.0 / self.top_n for sym in top.index}
                last_weights = dict(weights)
                last_rebalance_idx = i
                weights["timestamp"] = date
                records.append(weights)
            except Exception:
                continue
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records).set_index("timestamp").fillna(0.0)

    def _compute_returns(
        self,
        holdings: pd.DataFrame,
        prices: pd.DataFrame,
        dates: list,
        market_filter: pd.Series
    ) -> pd.DataFrame:
        daily_returns = []
        holding_dates = sorted(holdings.index)

        # 单币止损跟踪���{symbol: entry_price}
        entry_prices = {}
        # 被止损的币：当日清仓，次日不重新买入（等下次���子重新选）
        stopped_out = set()

        prev_holdings = pd.Series(dtype=float)

        for i, date in enumerate(holding_dates):
            # 市场过滤：熊市空仓
            if market_filter is not None and date in market_filter.index:
                if market_filter.loc[date] == 0:
                    prev_holdings = pd.Series(
                        {sym: 0.0 for sym in holdings.columns}
                    )
                    entry_prices = {}
                    stopped_out = set()
                    daily_returns.append({
                        "timestamp": date, "return": 0.0,
                        "gross_return": 0.0, "cost": 0.0, "turnover": 0.0
                    })
                    continue

            curr_desired = holdings.loc[date]

            if i == 0:
                prev_holdings = curr_desired.copy()
                # 记录买入价
                if date in prices.index:
                    for sym, w in prev_holdings.items():
                        if w > 0 and sym in prices.columns:
                            entry_prices[sym] = prices.loc[date, sym]
                continue

            prev_date = holding_dates[i - 1]
            if prev_date not in prices.index or date not in prices.index:
                continue

            price_today = prices.loc[date]
            price_prev = prices.loc[prev_date]

            # 检查单币止损
            new_stopped = set()
            for sym, entry_p in list(entry_prices.items()):
                if sym in price_today.index and entry_p > 0:
                    ret_from_entry = (price_today[sym] - entry_p) / entry_p
                    if ret_from_entry < self.stop_loss_single:
                        new_stopped.add(sym)
                        logger.debug(f"[StopLoss] {sym} at {date}, ret={ret_from_entry:.1%}")

            # 计算收益（用昨日持仓）
            portfolio_ret = 0.0
            for sym, w in prev_holdings.items():
                if w > 0 and sym not in stopped_out:
                    if sym in price_today.index and sym in price_prev.index:
                        pct = (price_today[sym] - price_prev[sym]) / (price_prev[sym] + 1e-10)
                        if not np.isnan(pct):
                            portfolio_ret += w * pct

            # 构建实际持仓（去掉止损的币，从因子重新选）
            actual_holdings = {}
            for sym, w in curr_desired.items():
                if w > 0 and sym not in new_stopped:
                    actual_holdings[sym] = w

            # 重���等权归一化
            if actual_holdings:
                total_w = sum(actual_holdings.values())
                actual_holdings = {s: w / total_w for s, w in actual_holdings.items()}

            actual_series = pd.Series(actual_holdings)

            # 换手成本
            turnover = self._calc_turnover(prev_holdings, actual_series)
            cost = turnover * self.cost_per_side * 2
            net_ret = portfolio_ret - cost

            daily_returns.append({
                "timestamp": date,
                "return": net_ret,
                "gross_return": portfolio_ret,
                "cost": cost,
                "turnover": turnover
            })

            # 更新持仓和止损记录
            prev_holdings = actual_series if not actual_series.empty else pd.Series(dtype=float)
            stopped_out = new_stopped  # 只在当日清仓，下次重选

            # 更新买入价（新进入的仓位）
            for sym in actual_holdings:
                if sym not in entry_prices or sym in new_stopped:
                    if sym in price_today.index:
                        entry_prices[sym] = price_today[sym]
            # 清除已止损的记录
            for sym in new_stopped:
                entry_prices.pop(sym, None)

        if not daily_returns:
            return pd.DataFrame()

        result_df = pd.DataFrame(daily_returns).set_index("timestamp")
        return result_df

    def _calc_turnover(self, prev: pd.Series, curr: pd.Series) -> float:
        all_syms = set(prev.index) | set(curr.index)
        turnover = sum(abs(curr.get(s, 0.0) - prev.get(s, 0.0)) for s in all_syms)
        return turnover / 2
