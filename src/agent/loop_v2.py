"""
Agent Loop v2 - 多因子合成 + 市场过滤
OUROBOROS 因子挖掘主循环（第二代）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger
from itertools import combinations
from typing import List

from factors.base import BaseFactor
from factors.momentum import (
    ReturnMomentum, RSIMomentum, MACDSignal,
    ShortTermReversal, PriceAcceleration
)
from factors.volume import (
    MoneyFlow, VolumeRatio, VWAP_Deviation, TradeIntensity
)
from factors.volatility import (
    HistoricalVolatility, ATR, BollingerBandWidth, DownsideVolatility
)
from factors.composite import CompositeFactor, MarketFilter
from backtest.engine import BacktestEngine
from backtest.metrics import full_metrics, print_metrics
from utils.memory import (
    record_success, record_failure, is_already_tried,
    append_iteration_log, update_iteration_header,
    load_factor_library, load_failed_factors
)

# 基础因子池（经过第一轮筛选，选 IC 相对较好的）
BASE_FACTORS = [
    ReturnMomentum(window=20),
    ReturnMomentum(window=60),
    RSIMomentum(window=21),
    MACDSignal(fast=12, slow=26, signal=9),
    MoneyFlow(window=10),
    VWAP_Deviation(window=10),
    TradeIntensity(window=10),
    HistoricalVolatility(window=20),
    ATR(window=14),
    BollingerBandWidth(window=20),
    DownsideVolatility(window=20),
]

# 组合策略配置
COMBO_CONFIGS = [
    # (因子列表, 描述, equal_weight)
    ([ReturnMomentum(20), RSIMomentum(21), MoneyFlow(10)], "mom+rsi+mf", True),
    ([ReturnMomentum(20), HistoricalVolatility(20), MoneyFlow(10)], "mom+hvol+mf", True),
    ([RSIMomentum(21), VWAP_Deviation(10), TradeIntensity(10)], "rsi+vwap+ti", True),
    ([ReturnMomentum(60), ATR(14), DownsideVolatility(20)], "mom60+atr+dvol", True),
    ([MoneyFlow(10), VWAP_Deviation(10), BollingerBandWidth(20)], "mf+vwap+bb", True),
    # IC加权版本
    ([ReturnMomentum(20), RSIMomentum(21), MoneyFlow(10)], "icw_mom+rsi+mf", False),
    ([ReturnMomentum(20), HistoricalVolatility(20), MoneyFlow(10)], "icw_mom+hvol+mf", False),
    # 全因子池合成
    (BASE_FACTORS[:6], "combo_6f_eq", True),
    (BASE_FACTORS, "combo_all_eq", True),
]


class OuroborosLoopV2:
    def __init__(
        self,
        panel: pd.DataFrame,
        train_start: str = "2020-01-01",
        train_end: str = "2023-12-31",
        test_start: str = "2024-01-01",
        test_end: str = None,
        top_n: int = 3,
        use_market_filter: bool = True,
    ):
        self.panel = panel
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.top_n = top_n
        self.use_market_filter = use_market_filter
        self.engine = BacktestEngine(top_n=top_n)
        self.market_filter = MarketFilter(ma_window=200)
        self.iteration = 0
        self.best_sharpe = 0.0

    def run(self):
        logger.info("[OUROBOROS v2] Multi-factor + Market Filter")
        logger.info(f"Train: {self.train_start} ~ {self.train_end}")
        logger.info(f"Test:  {self.test_start} ~ {self.test_end or 'latest'}")
        logger.info(f"Market filter: {'ON (BTC MA200)' if self.use_market_filter else 'OFF'}")

        # 计算市场过滤信号
        mf_signal = None
        if self.use_market_filter:
            try:
                mf_signal = self.market_filter.compute(self.panel)
                bull_pct = mf_signal.mean() * 100
                logger.info(f"Market filter: {bull_pct:.1f}% days in bull regime")
            except Exception as e:
                logger.warning(f"Market filter failed: {e}")

        for factors_list, name, equal_w in COMBO_CONFIGS:
            self.iteration += 1
            self._run_combo(factors_list, name, equal_w, mf_signal)

        self._final_report()

    def _run_combo(self, factors_list, name, equal_w, mf_signal):
        """测试一个多因子组合"""
        if is_already_tried(f"composite_{name}"):
            logger.info(f"[Skip] composite_{name} already tried")
            return

        logger.info(f"\n[Iter {self.iteration}] Composite: {name} (equal_weight={equal_w})")

        try:
            combo = CompositeFactor(factors_list, use_equal_weight=equal_w)
            factor = combo.compute(self.panel)
            factor.name = f"composite_{name}"
        except Exception as e:
            reason = f"Composite compute error: {e}"
            logger.error(reason)
            record_failure(f"composite_{name}", reason, {}, {"factors": name})
            return

        # 应用市场过滤：非牛市日期因子值置0
        if mf_signal is not None:
            factor = self._apply_market_filter(factor, mf_signal)

        # 样本内回测
        train_result = self.engine.run(
            factor, self.panel,
            start_date=self.train_start,
            end_date=self.train_end
        )
        if not isinstance(train_result, pd.DataFrame) or train_result.empty:
            reason = "Backtest returned empty"
            record_failure(f"composite_{name}", reason, {}, {})
            return

        train_returns = train_result["return"]
        train_metrics = full_metrics(
            train_returns, factor, self.panel,
            factor_name=f"composite_{name}_train"
        )
        print_metrics(train_metrics)

        # 样本外回测
        test_result = self.engine.run(
            factor, self.panel,
            start_date=self.test_start,
            end_date=self.test_end
        )
        test_metrics = {}
        if isinstance(test_result, pd.DataFrame) and not test_result.empty:
            test_returns = test_result["return"]
            test_metrics = full_metrics(
                test_returns, factor, self.panel,
                factor_name=f"composite_{name}_test"
            )
            print_metrics(test_metrics)

        # 记录结果
        passed = train_metrics.get("passed", False)
        oos_sharpe = test_metrics.get("sharpe", 0) if test_metrics else 0
        summary = self._build_summary(f"composite_{name}", train_metrics, test_metrics)

        if passed:
            record_success(train_metrics, "CompositeFactor", {"name": name, "equal_w": equal_w})
            if oos_sharpe > self.best_sharpe:
                self.best_sharpe = oos_sharpe
            logger.info(f"[✓] composite_{name} PASSED | OOS Sharpe: {oos_sharpe:.2f}")
        else:
            reason = self._diagnose(train_metrics)
            record_failure(f"composite_{name}", reason, train_metrics, {})
            logger.info(f"[✗] composite_{name} FAILED | {reason}")

        append_iteration_log(self.iteration, summary)
        lib = load_factor_library()
        failed = load_failed_factors()
        update_iteration_header(self.iteration, len(lib["factors"]), len(failed["factors"]))

    def _apply_market_filter(self, factor: pd.Series, mf_signal: pd.Series) -> pd.Series:
        """
        市场过滤：非牛市时将因子值置为 NaN（回测引擎会跳过 NaN）
        """
        filtered = factor.copy()
        dates = factor.index.get_level_values("timestamp").unique()
        for date in dates:
            if date in mf_signal.index:
                if mf_signal.loc[date] == 0:
                    # 熊市：清空因子信号
                    mask = filtered.index.get_level_values("timestamp") == date
                    filtered.loc[mask] = np.nan
        return filtered

    def _diagnose(self, metrics: dict) -> str:
        reasons = []
        if metrics.get("sharpe", 0) <= 1.5:
            reasons.append(f"Sharpe={metrics['sharpe']:.2f}<1.5")
        if metrics.get("max_drawdown", -1) < -0.20:
            reasons.append(f"MaxDD={metrics['max_drawdown']:.1%}>20%")
        if metrics.get("ic_mean", 0) <= 0.05:
            reasons.append(f"IC={metrics['ic_mean']:.4f}<0.05")
        return " | ".join(reasons) if reasons else "Unknown"

    def _build_summary(self, name, train_m, test_m) -> str:
        lines = [
            f"**因子组合**: {name}",
            f"**样本内** Sharpe={train_m.get('sharpe','N/A')} MaxDD={train_m.get('max_drawdown',0):.1%} IC={train_m.get('ic_mean',0):.4f}",
        ]
        if test_m:
            lines.append(f"**样本外** Sharpe={test_m.get('sharpe','N/A')} MaxDD={test_m.get('max_drawdown',0):.1%}")
        lines.append(f"**结论**: {'通过 ✓' if train_m.get('passed') else '失败 ✗'}")
        return "\n".join(lines)

    def _final_report(self):
        lib = load_factor_library()
        failed = load_failed_factors()
        logger.info(f"\n{'='*60}")
        logger.info(f"[OUROBOROS v2] 完成 {self.iteration} 次迭代")
        logger.info(f"���效因子组合: {len(lib['factors'])} 个")
        logger.info(f"失败: {len(failed['factors'])} 个")
        logger.info(f"最佳样本外 Sharpe: {self.best_sharpe:.2f}")
        if lib['factors']:
            for f in lib['factors']:
                m = f['metrics']
                logger.info(f"  {m['factor_name']}: Sharpe={m['sharpe']} IC={m['ic_mean']}")
        logger.info('='*60)


import numpy as np  # noqa - needed for _apply_market_filter
