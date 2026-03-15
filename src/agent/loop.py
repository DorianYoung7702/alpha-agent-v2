"""
Agent Loop v1 - 单因子挖掘主循环（修复版）
加入市场过滤（BTC MA200），放宽终止条件，让所有候选因子跑完
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger
from typing import List, Type

from factors.base import BaseFactor
from factors.momentum import (
    ReturnMomentum, RSIMomentum, MACDSignal,
    ShortTermReversal, PriceAcceleration,
    FundingRateReversal, FundingRateMomentum
)
from factors.volume import (
    VolumePriceDivergence, OBV, MoneyFlow,
    VolumeRatio, VWAP_Deviation, TradeIntensity
)
from factors.volatility import (
    HistoricalVolatility, ATR, BollingerBandWidth,
    VolatilityRatio, DownsideVolatility
)
from backtest.engine import BacktestEngine
from backtest.metrics import full_metrics, print_metrics
from utils.memory import (
    record_success, record_failure, is_already_tried,
    append_iteration_log, update_iteration_header,
    load_factor_library, load_failed_factors
)

# 全部候选因子（按 expert 建议优先级排序）
FACTOR_CANDIDATES = [
    # P0: 资金费率因子（���密独有 alpha，expert 预期 IC 0.05-0.10）
    (FundingRateReversal, {"window": 7}),
    (FundingRateReversal, {"window": 3}),
    (FundingRateReversal, {"window": 14}),
    (FundingRateMomentum, {"window": 7}),
    # P1: 动量因子（expert 建议用 21日，不是 5日）
    (ReturnMomentum, {"window": 21}),
    (ReturnMomentum, {"window": 14}),
    (ReturnMomentum, {"window": 7}),
    (ReturnMomentum, {"window": 30}),
    (ReturnMomentum, {"window": 60}),
    (RSIMomentum, {"window": 21}),
    (RSIMomentum, {"window": 14}),
    (RSIMomentum, {"window": 7}),
    (MACDSignal, {"fast": 12, "slow": 26, "signal": 9}),
    (MACDSignal, {"fast": 6, "slow": 13, "signal": 5}),
    (PriceAcceleration, {"short": 5, "long": 21}),
    # 量价类
    (MoneyFlow, {"window": 10}),
    (MoneyFlow, {"window": 5}),
    (VolumeRatio, {"window": 20}),
    (VWAP_Deviation, {"window": 10}),
    (TradeIntensity, {"window": 10}),
    (OBV, {"window": 20}),
    # 波动率类
    (HistoricalVolatility, {"window": 20}),
    (HistoricalVolatility, {"window": 10}),
    (ATR, {"window": 14}),
    (BollingerBandWidth, {"window": 20}),
    (DownsideVolatility, {"window": 20}),
]


def compute_market_filter(panel: pd.DataFrame, ma_window: int = 200) -> pd.Series:
    """
    BTC MA200 市场过滤：BTC 收盘 > MA200 = 牛市(1)，否则空仓(0)
    强制 shift(1) 防未来函数
    """
    try:
        btc = panel["close"].xs("BTCUSDT", level="symbol")
    except KeyError:
        syms = panel.index.get_level_values("symbol").unique()
        btc = panel["close"].xs(syms[0], level="symbol")
    ma = btc.rolling(ma_window, min_periods=ma_window // 2).mean()
    signal = (btc > ma).astype(float).shift(1)  # shift(1) 防未来函数
    bull_pct = signal.mean() * 100
    logger.info(f"[MarketFilter] BTC MA{ma_window}: {bull_pct:.1f}% days in bull regime")
    return signal


def apply_market_filter(factor: pd.Series, mf: pd.Series) -> pd.Series:
    """
    非牛市日期因子值置 NaN，回测引擎自动跳过空仓
    """
    filtered = factor.copy()
    dates = filtered.index.get_level_values("timestamp").unique()
    bear_dates = [d for d in dates if d in mf.index and mf.loc[d] == 0]
    for d in bear_dates:
        mask = filtered.index.get_level_values("timestamp") == d
        filtered.loc[mask] = np.nan
    bear_pct = len(bear_dates) / len(dates) * 100 if dates.size > 0 else 0
    logger.info(f"[MarketFilter] Filtered out {len(bear_dates)}/{len(dates)} bear days ({bear_pct:.1f}%)")
    return filtered


class OuroborosLoop:
    def __init__(
        self,
        panel: pd.DataFrame,
        train_start: str = "2020-01-01",
        train_end: str = "2023-12-31",
        test_start: str = "2024-01-01",
        test_end: str = None,
        top_n: int = 3,
        max_iterations: int = 100,
        max_no_improve: int = 50,  # 放宽，让所有因子都跑完
        use_market_filter: bool = True,
    ):
        self.panel = panel
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.top_n = top_n
        self.max_iterations = max_iterations
        self.max_no_improve = max_no_improve
        self.use_market_filter = use_market_filter
        self.engine = BacktestEngine(top_n=top_n)
        self.best_sharpe = 0.0
        self.no_improve_count = 0
        self.iteration = 0
        self.mf_signal = None

    def run(self):
        logger.info("[OUROBOROS] Starting single-factor mining")
        logger.info(f"Train: {self.train_start} ~ {self.train_end}")
        logger.info(f"Test:  {self.test_start} ~ {self.test_end or 'latest'}")
        logger.info(f"Market filter: {'ON (BTC MA200)' if self.use_market_filter else 'OFF'}")
        logger.info(f"Validation: Sharpe>1.5, IC>0.05 (no DD threshold)")

        if self.use_market_filter:
            self.mf_signal = compute_market_filter(self.panel)

        for factor_cls, params in FACTOR_CANDIDATES:
            if self.iteration >= self.max_iterations:
                logger.info("[OUROBOROS] Max iterations reached")
                break
            if self.no_improve_count >= self.max_no_improve:
                logger.info("[OUROBOROS] No improvement streak too long, stopping")
                break
            self.iteration += 1
            self._run_one(factor_cls, params)

        self._final_report()

    def _run_one(self, factor_cls: Type[BaseFactor], params: dict):
        try:
            factor_obj = factor_cls(**params)
        except Exception as e:
            logger.error(f"Factor init error: {e}")
            return

        factor_name = factor_obj.name
        if is_already_tried(factor_name):
            logger.info(f"[Skip] {factor_name} already tried")
            return

        logger.info(f"\n[Iter {self.iteration}] Testing: {factor_name}")

        # 计算因子（含 shift+标准化）
        try:
            factor = factor_obj.compute(self.panel)
        except Exception as e:
            reason = f"Compute error: {e}"
            logger.error(reason)
            record_failure(factor_name, reason, {}, params)
            self.no_improve_count += 1
            return

        # 市场过滤
        if self.mf_signal is not None:
            factor = apply_market_filter(factor, self.mf_signal)

        # 样本内回测
        train_result = self.engine.run(
            factor, self.panel,
            start_date=self.train_start,
            end_date=self.train_end
        )
        if not isinstance(train_result, pd.DataFrame) or train_result.empty:
            reason = "Backtest empty"
            record_failure(factor_name, reason, {}, params)
            self.no_improve_count += 1
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

        # 判断
        passed = train_metrics.get("passed", False)
        oos_sharpe = test_metrics.get("sharpe", 0) if test_metrics else 0
        summary = self._build_summary(factor_name, train_metrics, test_metrics, params)

        if passed:
            record_success(train_metrics, factor_cls.__name__, params)
            if oos_sharpe > self.best_sharpe:
                self.best_sharpe = oos_sharpe
                self.no_improve_count = 0
                logger.info(f"[★] New best OOS Sharpe: {oos_sharpe:.2f}")
            else:
                self.no_improve_count += 1
            logger.info(f"[✓] {factor_name} PASSED | OOS Sharpe: {oos_sharpe:.2f}")
        else:
            reason = self._diagnose(train_metrics)
            record_failure(factor_name, reason, train_metrics, params)
            self.no_improve_count += 1
            logger.info(f"[✗] {factor_name} FAILED | {reason}")

        append_iteration_log(self.iteration, summary)
        lib = load_factor_library()
        failed = load_failed_factors()
        update_iteration_header(self.iteration, len(lib["factors"]), len(failed["factors"]))

    def _diagnose(self, metrics: dict) -> str:
        reasons = []
        if metrics.get("sharpe", 0) <= 1.5:
            reasons.append(f"Sharpe={metrics['sharpe']:.2f}<1.5")
        if metrics.get("ic_mean", 0) <= 0.05:
            reasons.append(f"IC={metrics['ic_mean']:.4f}<0.05")
        return " | ".join(reasons) if reasons else "Unknown"

    def _build_summary(self, name, train_m, test_m, params) -> str:
        lines = [
            f"**因子**: {name} | 参数: {params}",
            f"**样本内** Sharpe={train_m.get('sharpe','N/A')} MaxDD={train_m.get('max_drawdown',0):.1%} IC={train_m.get('ic_mean',0):.4f} ICIR={train_m.get('icir',0):.4f}",
        ]
        if test_m:
            lines.append(
                f"**样本外** Sharpe={test_m.get('sharpe','N/A')} MaxDD={test_m.get('max_drawdown',0):.1%} IC={test_m.get('ic_mean',0):.4f}"
            )
        lines.append(f"**结论**: {'通过 ✓' if train_m.get('passed') else '���败 ✗'}")
        if not train_m.get('passed'):
            lines.append(f"**原因**: {self._diagnose(train_m)}")
        return "\n".join(lines)

    def _final_report(self):
        lib = load_factor_library()
        failed = load_failed_factors()
        logger.info(f"\n{'='*60}")
        logger.info(f"[OUROBOROS] 完成 {self.iteration} 次迭代")
        logger.info(f"有效因子: {len(lib['factors'])} 个")
        logger.info(f"失败因子: {len(failed['factors'])} ���")
        logger.info(f"最佳样本外 Sharpe: {self.best_sharpe:.2f}")
        if lib['factors']:
            logger.info("[通过的因子]")
            for f in lib['factors']:
                m = f['metrics']
                logger.info(f"  {m['factor_name']}: Sharpe={m['sharpe']} IC={m['ic_mean']}")
        logger.info('='*60)
