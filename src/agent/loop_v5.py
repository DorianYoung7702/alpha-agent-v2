"""
Agent Loop v5 - 时序趋势跟踪
单币 BTC/ETH 择时，不做截面选币
验证标准：Sharpe>1.5, MaxDD<40%, 样本外Sharpe>0.8
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from factors.ts_signals import (
    signal_macd, signal_bollinger, signal_ma_trend,
    signal_ma_cross, signal_rsi_trend, signal_combined
)
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics, print_metrics
from utils.memory import (
    record_success, record_failure, is_already_tried,
    append_iteration_log, update_iteration_header,
    load_factor_library, load_failed_factors
)

SYMBOLS = ["BTCUSDT", "ETHUSDT"]

SIGNAL_CONFIGS = [
    ("macd_12_26_9", signal_macd, {"fast": 12, "slow": 26, "signal": 9}),
    ("macd_6_13_5", signal_macd, {"fast": 6, "slow": 13, "signal": 5}),
    ("macd_19_39_9", signal_macd, {"fast": 19, "slow": 39, "signal": 9}),
    ("bb_20_2", signal_bollinger, {"window": 20, "n_std": 2.0}),
    ("bb_20_1p5", signal_bollinger, {"window": 20, "n_std": 1.5}),
    ("ma_trend_20_50_200", signal_ma_trend, {"fast": 20, "slow": 50, "trend": 200}),
    ("ma_cross_20_60", signal_ma_cross, {"fast": 20, "slow": 60}),
    ("ma_cross_10_30", signal_ma_cross, {"fast": 10, "slow": 30}),
    ("ma_cross_50_200", signal_ma_cross, {"fast": 50, "slow": 200}),
    ("rsi_trend_14", signal_rsi_trend, {"rsi_window": 14}),
    ("rsi_trend_21", signal_rsi_trend, {"rsi_window": 21}),
    ("combined_majority", signal_combined, {"require_all": False}),
    ("combined_all", signal_combined, {"require_all": True}),
]


class OuroborosLoopV5:
    def __init__(
        self,
        data: dict,
        train_start: str = "2020-01-01",
        train_end: str = "2023-12-31",
        test_start: str = "2024-01-01",
        test_end: str = None,
        oos_sharpe_threshold: float = 0.8,
    ):
        self.data = data
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.oos_threshold = oos_sharpe_threshold
        self.engine = TimeSeriesEngine(atr_mult=2.0, use_atr_stop=True)
        self.iteration = 0
        self.best_oos = 0.0
        self.passed_strategies = []

    def run(self):
        logger.info("[OUROBOROS v5] Time-Series Trend Following")
        logger.info(f"Symbols: {SYMBOLS}")
        logger.info(f"Train: {self.train_start} ~ {self.train_end}")
        logger.info(f"Test:  {self.test_start} ~ {self.test_end or 'latest'}")
        logger.info(f"Validation: Sharpe>1.5, MaxDD<40%, OOS Sharpe>{self.oos_threshold}")

        for symbol in SYMBOLS:
            if symbol not in self.data:
                logger.warning(f"No data for {symbol}")
                continue
            ohlcv = self.data[symbol]
            logger.info(f"\n{'='*50}\nTesting {symbol} ({len(ohlcv)} days)")
            for sig_name, sig_func, sig_params in SIGNAL_CONFIGS:
                self.iteration += 1
                self._run_one(symbol, ohlcv, sig_name, sig_func, sig_params)

        self._final_report()

    def _run_one(self, symbol, ohlcv, sig_name, sig_func, sig_params):
        strategy_name = f"{symbol}_{sig_name}"
        if is_already_tried(strategy_name):
            logger.info(f"[Skip] {strategy_name}")
            return

        logger.info(f"\n[Iter {self.iteration}] {strategy_name}")
        try:
            signal = sig_func(ohlcv, **sig_params)
        except Exception as e:
            record_failure(strategy_name, f"Signal error: {e}", {}, sig_params)
            return

        train_result = self.engine.run(signal, ohlcv,
            start_date=self.train_start, end_date=self.train_end)
        if not isinstance(train_result, pd.DataFrame) or train_result.empty:
            record_failure(strategy_name, "Empty train", {}, sig_params)
            return

        train_metrics = full_metrics(train_result["return"],
            factor_name=f"{strategy_name}_train")
        print_metrics(train_metrics)

        test_metrics = {}
        test_result = self.engine.run(signal, ohlcv,
            start_date=self.test_start, end_date=self.test_end)
        if isinstance(test_result, pd.DataFrame) and not test_result.empty:
            test_metrics = full_metrics(test_result["return"],
                factor_name=f"{strategy_name}_test")
            print_metrics(test_metrics)

        train_passed = train_metrics.get("passed", False)
        oos_sharpe = test_metrics.get("sharpe", 0) if test_metrics else 0
        oos_dd = test_metrics.get("max_drawdown", -1) if test_metrics else -1
        oos_passed = oos_sharpe > self.oos_threshold and oos_dd > -0.40

        summary = (
            f"**{strategy_name}**\n"
            f"样本内 Sharpe={train_metrics.get('sharpe')} "
            f"MaxDD={train_metrics.get('max_drawdown',0):.1%}\n"
            + (f"样本外 Sharpe={test_metrics.get('sharpe')} "
               f"MaxDD={test_metrics.get('max_drawdown',0):.1%}\n" if test_metrics else "")
            + f"结论: {'双通过 ✓✓' if (train_passed and oos_passed) else '样本内 ✓' if train_passed else '失败 ✗'}"
        )

        if train_passed and oos_passed:
            record_success(train_metrics, sig_name, {"symbol": symbol, **sig_params})
            if oos_sharpe > self.best_oos:
                self.best_oos = oos_sharpe
            self.passed_strategies.append(strategy_name)
            logger.info(f"[✓✓] {strategy_name} PASSED | OOS Sharpe={oos_sharpe:.2f} MaxDD={oos_dd:.1%}")
        elif train_passed:
            record_failure(strategy_name, f"OOS Sharpe={oos_sharpe:.2f}<{self.oos_threshold}", train_metrics, sig_params)
            logger.info(f"[✓✗] {strategy_name} train OK but OOS failed | OOS={oos_sharpe:.2f}")
        else:
            reasons = []
            if train_metrics.get('sharpe', 0) <= 1.5:
                reasons.append(f"Sharpe={train_metrics['sharpe']:.2f}<1.5")
            if train_metrics.get('max_drawdown', -1) < -0.40:
                reasons.append(f"MaxDD={train_metrics['max_drawdown']:.1%}>40%")
            record_failure(strategy_name, " | ".join(reasons), train_metrics, sig_params)
            logger.info(f"[✗] {strategy_name} FAILED | {' | '.join(reasons)}")

        append_iteration_log(self.iteration, summary)
        lib = load_factor_library()
        failed = load_failed_factors()
        update_iteration_header(self.iteration, len(lib["factors"]), len(failed["factors"]))

    def _final_report(self):
        lib = load_factor_library()
        failed = load_failed_factors()
        logger.info(f"\n{'='*60}")
        logger.info(f"[OUROBOROS v5] 完成 {self.iteration} 次迭代")
        logger.info(f"通过策略: {len(self.passed_strategies)} 个")
        logger.info(f"失败: {len(failed['factors'])} 个")
        logger.info(f"最佳样本外 Sharpe: {self.best_oos:.2f}")
        for s in self.passed_strategies:
            logger.info(f"  ✓✓ {s}")
        logger.info('='*60)
