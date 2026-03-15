"""
Agent Loop v4 - IC滚动过滤 + Top8 + 3日换仓
专注验证 hvol_20d 和 downvol_20d
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from scipy import stats
from loguru import logger

from factors.volatility import HistoricalVolatility, DownsideVolatility, ATR, BollingerBandWidth
from factors.momentum import ReturnMomentum, RSIMomentum, FundingRateReversal
from factors.volume import MoneyFlow
from backtest.engine import BacktestEngine
from backtest.metrics import full_metrics, print_metrics
from utils.memory import (
    record_success, record_failure, is_already_tried,
    append_iteration_log, update_iteration_header,
    load_factor_library, load_failed_factors
)


def compute_ic_filter(
    factor: pd.Series,
    panel: pd.DataFrame,
    ic_window: int = 30,
    ic_threshold: float = 0.0
) -> pd.Series:
    """
    IC 滚动过滤：计算每日滚动 IC ���值
    返回布尔 Series：True = 过去 ic_window 日均 IC > threshold，允许交易
    强制 shift(1) 防未���函数
    """
    close = panel["close"].unstack(level="symbol")
    fwd_ret = (close.shift(-1) / close - 1).stack()
    fwd_ret.index.names = ["timestamp", "symbol"]

    combined = pd.concat(
        [factor.rename("factor"), fwd_ret.rename("fwd_ret")],
        axis=1
    ).dropna()

    def ic_row(g):
        if len(g) < 5:
            return np.nan
        c, _ = stats.spearmanr(g["factor"], g["fwd_ret"])
        return c

    daily_ic = combined.groupby(level="timestamp").apply(ic_row).dropna()
    rolling_ic = daily_ic.rolling(ic_window, min_periods=ic_window // 2).mean()
    # shift(1): 用昨日的滚动IC判断今日是否交易
    trade_signal = (rolling_ic.shift(1) > ic_threshold)
    logger.info(f"[IC Filter] Active days: {trade_signal.sum()}/{len(trade_signal)} ({trade_signal.mean():.1%})")
    return trade_signal


def apply_ic_filter(factor: pd.Series, trade_signal: pd.Series) -> pd.Series:
    """将 IC 过滤应用到因子：不允许交易的日期置 NaN"""
    filtered = factor.copy()
    dates = filtered.index.get_level_values("timestamp").unique()
    blocked = [d for d in dates if d in trade_signal.index and not trade_signal.loc[d]]
    for d in blocked:
        mask = filtered.index.get_level_values("timestamp") == d
        filtered.loc[mask] = np.nan
    logger.info(f"[IC Filter] Blocked {len(blocked)}/{len(dates)} days ({len(blocked)/len(dates):.1%})")
    return filtered


# 第五轮候选因���（聚焦高IC因子 + 参数变体）
FACTOR_CANDIDATES = [
    # 最高IC因子
    (HistoricalVolatility, {"window": 20}, "v4"),
    (DownsideVolatility, {"window": 20}, "v4"),
    (HistoricalVolatility, {"window": 10}, "v4"),
    (HistoricalVolatility, {"window": 30}, "v4"),
    (DownsideVolatility, {"window": 10}, "v4"),
    # 资金费率
    (FundingRateReversal, {"window": 7}, "v4"),
    (FundingRateReversal, {"window": 14}, "v4"),
    # 其他
    (ReturnMomentum, {"window": 21}, "v4"),
    (MoneyFlow, {"window": 10}, "v4"),
    (RSIMomentum, {"window": 21}, "v4"),
]


class OuroborosLoopV4:
    def __init__(
        self,
        panel: pd.DataFrame,
        train_start: str = "2020-01-01",
        train_end: str = "2023-12-31",
        test_start: str = "2024-01-01",
        test_end: str = None,
        top_n: int = 8,
        rebalance_days: int = 3,
        use_ic_filter: bool = True,
        ic_window: int = 30,
        ic_threshold: float = 0.0,
        use_market_filter: bool = True,
    ):
        self.panel = panel
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.use_ic_filter = use_ic_filter
        self.ic_window = ic_window
        self.ic_threshold = ic_threshold
        self.use_market_filter = use_market_filter
        self.engine = BacktestEngine(
            top_n=top_n,
            rebalance_every_n_days=rebalance_days
        )
        self.mf_signal = None
        self.iteration = 0
        self.best_oos_sharpe = 0.0

    def run(self):
        logger.info("[OUROBOROS v4] Top8 + 3日换仓 + IC滚动过滤")
        logger.info(f"Top_N={self.engine.top_n}, Rebalance={self.engine.rebalance_every_n_days}日")
        logger.info(f"IC Filter: {'ON' if self.use_ic_filter else 'OFF'} (window={self.ic_window}, threshold={self.ic_threshold})")

        if self.use_market_filter:
            try:
                btc = self.panel["close"].xs("BTCUSDT", level="symbol")
                ma200 = btc.rolling(200).mean()
                self.mf_signal = (btc > ma200).astype(float).shift(1)
                logger.info(f"[MarketFilter] BTC MA200: {self.mf_signal.mean():.1%} days bull")
            except Exception as e:
                logger.warning(f"Market filter failed: {e}")

        for factor_cls, params, suffix in FACTOR_CANDIDATES:
            self.iteration += 1
            self._run_one(factor_cls, params, suffix)

        self._final_report()

    def _run_one(self, factor_cls, params, suffix):
        try:
            factor_obj = factor_cls(**params)
        except Exception as e:
            logger.error(f"Init error: {e}")
            return

        factor_name = f"{factor_obj.name}_{suffix}"
        if is_already_tried(factor_name):
            logger.info(f"[Skip] {factor_name}")
            return

        logger.info(f"\n[Iter {self.iteration}] {factor_name}")

        try:
            factor = factor_obj.compute(self.panel)
        except Exception as e:
            reason = f"Compute error: {e}"
            logger.error(reason)
            record_failure(factor_name, reason, {}, params)
            return

        # 应用 IC 滚动过滤
        if self.use_ic_filter:
            try:
                trade_signal = compute_ic_filter(
                    factor, self.panel,
                    ic_window=self.ic_window,
                    ic_threshold=self.ic_threshold
                )
                factor = apply_ic_filter(factor, trade_signal)
            except Exception as e:
                logger.warning(f"IC filter failed: {e}")

        # 应用市场过滤
        if self.mf_signal is not None:
            dates = factor.index.get_level_values("timestamp").unique()
            bear_dates = [d for d in dates if d in self.mf_signal.index and self.mf_signal.loc[d] == 0]
            for d in bear_dates:
                mask = factor.index.get_level_values("timestamp") == d
                factor.loc[mask] = np.nan

        # 样本内回测
        train_result = self.engine.run(
            factor, self.panel,
            start_date=self.train_start,
            end_date=self.train_end
        )
        if not isinstance(train_result, pd.DataFrame) or train_result.empty:
            record_failure(factor_name, "Empty train backtest", {}, params)
            return

        train_returns = train_result["return"]
        train_metrics = full_metrics(
            train_returns, factor, self.panel,
            factor_name=f"{factor_name}_train"
        )
        print_metrics(train_metrics)

        # 样本外回测
        test_metrics = {}
        test_result = self.engine.run(
            factor, self.panel,
            start_date=self.test_start,
            end_date=self.test_end
        )
        if isinstance(test_result, pd.DataFrame) and not test_result.empty:
            test_returns = test_result["return"]
            test_metrics = full_metrics(
                test_returns, factor, self.panel,
                factor_name=f"{factor_name}_test"
            )
            print_metrics(test_metrics)

        passed = train_metrics.get("passed", False)
        oos_sharpe = test_metrics.get("sharpe", 0) if test_metrics else 0
        oos_dd = test_metrics.get("max_drawdown", -1) if test_metrics else -1

        summary = (
            f"**{factor_name}** | Top{self.engine.top_n} | {self.engine.rebalance_every_n_days}日换仓\n"
            f"样���内 Sharpe={train_metrics.get('sharpe')} MaxDD={train_metrics.get('max_drawdown',0):.1%} "
            f"IC={train_metrics.get('ic_mean',0):.4f} ICIR={train_metrics.get('icir',0):.4f}\n"
            + (f"样本外 Sharpe={test_metrics.get('sharpe')} MaxDD={test_metrics.get('max_drawdown',0):.1%}\n"
               if test_metrics else "")
            + f"结论: {'���过 ✓' if passed else '失败 ✗'}"
        )

        if passed:
            record_success(train_metrics, factor_cls.__name__, params)
            if oos_sharpe > self.best_oos_sharpe:
                self.best_oos_sharpe = oos_sharpe
            logger.info(f"[✓] {factor_name} PASSED | OOS Sharpe={oos_sharpe:.2f} MaxDD={oos_dd:.1%}")
        else:
            reasons = []
            if train_metrics.get('sharpe', 0) <= 1.5:
                reasons.append(f"Sharpe={train_metrics['sharpe']:.2f}<1.5")
            if train_metrics.get('max_drawdown', -1) < -0.20:
                reasons.append(f"MaxDD={train_metrics['max_drawdown']:.1%}>20%")
            if train_metrics.get('ic_mean', 0) <= 0.05:
                reasons.append(f"IC={train_metrics['ic_mean']:.4f}<0.05")
            reason = " | ".join(reasons) if reasons else "Unknown"
            record_failure(factor_name, reason, train_metrics, params)
            logger.info(f"[✗] {factor_name} FAILED | {reason}")

        append_iteration_log(self.iteration, summary)
        lib = load_factor_library()
        failed = load_failed_factors()
        update_iteration_header(self.iteration, len(lib["factors"]), len(failed["factors"]))

    def _final_report(self):
        lib = load_factor_library()
        failed = load_failed_factors()
        logger.info(f"\n{'='*60}")
        logger.info(f"[OUROBOROS v4] 完成 {self.iteration} 次迭代")
        logger.info(f"有效因子: {len(lib['factors'])} 个")
        logger.info(f"失败: {len(failed['factors'])} 个")
        logger.info(f"最佳样本外 Sharpe: {self.best_oos_sharpe:.2f}")
        if lib['factors']:
            for f in lib['factors']:
                m = f['metrics']
                logger.info(f"  {m['factor_name']}: Sharpe={m['sharpe']} MaxDD={m['max_drawdown']:.1%} IC={m['ic_mean']}")
        logger.info('='*60)
