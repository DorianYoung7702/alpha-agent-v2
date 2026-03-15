"""
Agent Loop v5b - 时序趋势跟踪优化版
基于 BTCUSDT_combined_all (Sharpe=0.94, MaxDD=-38.1%) 进行参数优化
目标：Sharpe>1.5, MaxDD<40%
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
from factors.ts_signals_v2 import (
    signal_combined_adx, signal_ma_cross_adx,
    signal_triple_ma, signal_macd_ma_filter
)
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics, print_metrics
from utils.memory import (
    record_success, record_failure, is_already_tried,
    append_iteration_log, update_iteration_header,
    load_factor_library, load_failed_factors
)

SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# 优化配置：聚焦 combined_all 的变体 + 新信号
SIGNAL_CONFIGS = [
    # combined_all 变体（不同ATR止损）
    ("combined_adx20_all", signal_combined_adx, {"adx_threshold": 20.0, "require_all": True}),
    ("combined_adx25_all", signal_combined_adx, {"adx_threshold": 25.0, "require_all": True}),
    ("combined_adx15_all", signal_combined_adx, {"adx_threshold": 15.0, "require_all": True}),
    ("combined_adx20_maj", signal_combined_adx, {"adx_threshold": 20.0, "require_all": False}),
    # 均线交叉 + ADX
    ("ma_cross_adx20_20_60", signal_ma_cross_adx, {"fast": 20, "slow": 60, "adx_threshold": 20.0}),
    ("ma_cross_adx25_20_60", signal_ma_cross_adx, {"fast": 20, "slow": 60, "adx_threshold": 25.0}),
    ("ma_cross_adx20_50_200", signal_ma_cross_adx, {"fast": 50, "slow": 200, "adx_threshold": 20.0}),
    # 三线排列
    ("triple_ma_10_30_100", signal_triple_ma, {"fast": 10, "mid": 30, "slow": 100}),
    ("triple_ma_20_50_200", signal_triple_ma, {"fast": 20, "mid": 50, "slow": 200}),
    ("triple_ma_5_20_60", signal_triple_ma, {"fast": 5, "mid": 20, "slow": 60}),
    # MACD + MA过滤
    ("macd_ma200", signal_macd_ma_filter, {"fast": 12, "slow": 26, "signal_period": 9, "ma_filter": 200}),
    ("macd_ma100", signal_macd_ma_filter, {"fast": 12, "slow": 26, "signal_period": 9, "ma_filter": 100}),
]

# 测试不同 ATR 止损参���
ATR_CONFIGS = [
    {"atr_mult": 1.5, "use_atr_stop": True},
    {"atr_mult": 2.0, "use_atr_stop": True},
    {"atr_mult": 3.0, "use_atr_stop": True},
    {"atr_mult": 2.0, "use_atr_stop": False},  # 无止损对照
]


class OuroborosLoopV5b:
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
        self.iteration = 0
        self.best_oos = 0.0
        self.passed_strategies = []

    def run(self):
        logger.info("[OUROBOROS v5b] Optimizing combined_all + new signals")
        logger.info(f"Testing {len(SIGNAL_CONFIGS)} signals × {len(ATR_CONFIGS)} ATR configs × {len(SYMBOLS)} symbols")

        for symbol in SYMBOLS:
            if symbol not in self.data:
                continue
            ohlcv = self.data[symbol]
            logger.info(f"\n{'='*50}\n{symbol}")

            for sig_name, sig_func, sig_params in SIGNAL_CONFIGS:
                for atr_cfg in ATR_CONFIGS:
                    self.iteration += 1
                    atr_tag = f"atr{atr_cfg['atr_mult']}" if atr_cfg['use_atr_stop'] else "nostop"
                    strategy_name = f"{symbol}_{sig_name}_{atr_tag}"
                    self._run_one(symbol, ohlcv, strategy_name, sig_func, sig_params, atr_cfg)

        self._final_report()

    def _run_one(self, symbol, ohlcv, strategy_name, sig_func, sig_params, atr_cfg):
        if is_already_tried(strategy_name):
            logger.info(f"[Skip] {strategy_name}")
            return

        logger.info(f"[Iter {self.iteration}] {strategy_name}")
        engine = TimeSeriesEngine(
            atr_mult=atr_cfg["atr_mult"],
            use_atr_stop=atr_cfg["use_atr_stop"]
        )

        try:
            signal = sig_func(ohlcv, **sig_params)
        except Exception as e:
            record_failure(strategy_name, f"Signal error: {e}", {}, sig_params)
            return

        train_result = engine.run(signal, ohlcv,
            start_date=self.train_start, end_date=self.train_end)
        if not isinstance(train_result, pd.DataFrame) or train_result.empty:
            record_failure(strategy_name, "Empty train", {}, sig_params)
            return

        train_metrics = full_metrics(train_result["return"],
            factor_name=f"{strategy_name}_train")

        test_metrics = {}
        test_result = engine.run(signal, ohlcv,
            start_date=self.test_start, end_date=self.test_end)
        if isinstance(test_result, pd.DataFrame) and not test_result.empty:
            test_metrics = full_metrics(test_result["return"],
                factor_name=f"{strategy_name}_test")

        train_passed = train_metrics.get("passed", False)
        oos_sharpe = test_metrics.get("sharpe", 0) if test_metrics else 0
        oos_dd = test_metrics.get("max_drawdown", -1) if test_metrics else -1
        oos_passed = oos_sharpe > self.oos_threshold and oos_dd > -0.40

        # 只打印通过的
        if train_passed:
            print_metrics(train_metrics)
            if test_metrics:
                print_metrics(test_metrics)

        summary = (
            f"**{strategy_name}**\n"
            f"样本内 Sharpe={train_metrics.get('sharpe')} MaxDD={train_metrics.get('max_drawdown',0):.1%}\n"
            + (f"样本外 Sharpe={test_metrics.get('sharpe')} MaxDD={test_metrics.get('max_drawdown',0):.1%}\n" if test_metrics else "")
            + f"结论: {'双通过 ✓✓' if (train_passed and oos_passed) else '样本内 ✓' if train_passed else '失败 ✗'}"
        )

        if train_passed and oos_passed:
            record_success(train_metrics, str(sig_func.__name__), {"symbol": symbol, **sig_params, **atr_cfg})
            if oos_sharpe > self.best_oos:
                self.best_oos = oos_sharpe
            self.passed_strategies.append(strategy_name)
            logger.info(f"[✓✓] {strategy_name} PASSED | OOS={oos_sharpe:.2f} MaxDD={oos_dd:.1%}")
        elif train_passed:
            record_failure(strategy_name, f"OOS={oos_sharpe:.2f}<{self.oos_threshold}", train_metrics, sig_params)
            logger.info(f"[✓✗] {strategy_name} train OK | OOS={oos_sharpe:.2f} MaxDD={oos_dd:.1%}")
        else:
            s = train_metrics.get('sharpe', 0)
            dd = train_metrics.get('max_drawdown', -1)
            reasons = []
            if s <= 1.5: reasons.append(f"Sharpe={s:.2f}")
            if dd < -0.40: reasons.append(f"MaxDD={dd:.1%}")
            record_failure(strategy_name, " | ".join(reasons), train_metrics, sig_params)
            logger.debug(f"[✗] {strategy_name} | {' | '.join(reasons)}")

        append_iteration_log(self.iteration, summary)
        lib = load_factor_library()
        failed = load_failed_factors()
        update_iteration_header(self.iteration, len(lib["factors"]), len(failed["factors"]))

    def _final_report(self):
        lib = load_factor_library()
        failed = load_failed_factors()
        logger.info(f"\n{'='*60}")
        logger.info(f"[OUROBOROS v5b] 完成 {self.iteration} 次迭代")
        logger.info(f"通过策略: {len(self.passed_strategies)} 个")
        logger.info(f"失败: {len(failed['factors'])} 个")
        logger.info(f"最佳样本外 Sharpe: {self.best_oos:.2f}")
        for s in self.passed_strategies:
            logger.info(f"  ✓✓ {s}")
        # 打印样本内最佳结果
        all_results = [(f['factor_name'], f.get('metrics', {}).get('sharpe', 0),
                        f.get('metrics', {}).get('max_drawdown', -1))
                       for f in failed['factors']]
        all_results.sort(key=lambda x: x[1], reverse=True)
        logger.info("\nTop10 样本内 Sharpe:")
        for name, sh, dd in all_results[:10]:
            logger.info(f"  {name}: Sharpe={sh:.2f} MaxDD={dd:.1%}")
        logger.info('='*60)
