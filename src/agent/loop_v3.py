"""
Agent Loop v3 - 市场中性策略 + Agent协同
做多 Top3 + 做空 Bottom3，天然控制回撤
每轮���测后自动通知 Expert 分析，通知秘书记录
"""
import sys
import json
import subprocess
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from factors.momentum import ReturnMomentum, RSIMomentum, MACDSignal, PriceAcceleration
from factors.volume import MoneyFlow, VolumeRatio, VWAP_Deviation, TradeIntensity, OBV
from factors.volatility import HistoricalVolatility, ATR, BollingerBandWidth, DownsideVolatility
from backtest.engine_neutral import MarketNeutralEngine
from backtest.metrics import full_metrics, print_metrics
from utils.memory import (
    record_success, record_failure, is_already_tried,
    append_iteration_log, update_iteration_header,
    load_factor_library, load_failed_factors
)

FACTOR_CANDIDATES = [
    (HistoricalVolatility, {"window": 20}),
    (HistoricalVolatility, {"window": 10}),
    (RSIMomentum, {"window": 21}),
    (RSIMomentum, {"window": 14}),
    (MoneyFlow, {"window": 10}),
    (ReturnMomentum, {"window": 20}),
    (RSIMomentum, {"window": 7}),
    (ReturnMomentum, {"window": 60}),
    (ReturnMomentum, {"window": 5}),
    (ReturnMomentum, {"window": 10}),
    (MACDSignal, {"fast": 12, "slow": 26, "signal": 9}),
    (MACDSignal, {"fast": 6, "slow": 13, "signal": 5}),
    (MoneyFlow, {"window": 5}),
    (VWAP_Deviation, {"window": 10}),
    (VWAP_Deviation, {"window": 5}),
    (TradeIntensity, {"window": 10}),
    (OBV, {"window": 20}),
    (ATR, {"window": 14}),
    (BollingerBandWidth, {"window": 20}),
    (DownsideVolatility, {"window": 20}),
    (PriceAcceleration, {"short": 5, "long": 20}),
]

OPENCLAW_CMD = r"D:\npm-global\openclaw.cmd"

def _send_to_agent(session_key: str, message: str):
    """通过 OpenClaw CLI 给指定 agent 发消���"""
    try:
        result = subprocess.run(
            [OPENCLAW_CMD, "message", "send",
             "--session", session_key,
             "--message", message],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            logger.info(f"[Agent通信] 消息已发送到 {session_key}")
        else:
            logger.warning(f"[Agent通信] 发送失败: {result.stderr}")
    except Exception as e:
        logger.warning(f"[Agent通信] 异常: {e}")


class OuroborosLoopV3:
    def __init__(
        self,
        panel: pd.DataFrame,
        train_start: str = "2020-01-01",
        train_end: str = "2023-12-31",
        test_start: str = "2024-01-01",
        test_end: str = None,
        top_n: int = 3,
    ):
        self.panel = panel
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.top_n = top_n
        self.engine = MarketNeutralEngine(top_n=top_n)
        self.best_sharpe = 0.0
        self.iteration = 0
        self._last_expert_check = 0  # 上次读取专家建议的迭代号

    def _reload_expert_guidance(self):
        """每轮迭代前重新读取专家建议"""
        guidance_path = Path(__file__).parent.parent.parent / "memory" / "expert_guidance.md"
        if guidance_path.exists():
            content = guidance_path.read_text(encoding="utf-8")
            logger.info(f"[Expert] 已读取最新专家建议（{len(content)}字）")
        else:
            logger.info("[Expert] 暂无专家建议")

    def run(self):
        logger.info("[OUROBOROS v3] Market Neutral Strategy")
        logger.info(f"Long Top{self.top_n} + Short Bottom{self.top_n}")
        logger.info(f"Train: {self.train_start} ~ {self.train_end}")
        logger.info(f"Test:  {self.test_start} ~ {self.test_end or 'latest'}")

        # 读取专家建议
        self._reload_expert_guidance()

        for factor_cls, params in FACTOR_CANDIDATES:
            self.iteration += 1
            # 每轮前重新读取专家建议
            self._reload_expert_guidance()
            self._run_one(factor_cls, params)

        self._final_report()

    def _run_one(self, factor_cls, params):
        try:
            factor_obj = factor_cls(**params)
        except Exception as e:
            logger.error(f"Init error: {e}")
            return

        factor_name = f"mn_{factor_obj.name}"
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

        train_result = self.engine.run(
            factor, self.panel,
            start_date=self.train_start,
            end_date=self.train_end
        )
        if not isinstance(train_result, pd.DataFrame) or train_result.empty:
            record_failure(factor_name, "Empty backtest", {}, params)
            return

        train_returns = train_result["return"]
        train_metrics = full_metrics(
            train_returns, factor, self.panel,
            factor_name=f"{factor_name}_train"
        )
        print_metrics(train_metrics)

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
        summary = (
            f"**{factor_name}** | 参数:{params}\n"
            f"样本内 Sharpe={train_metrics.get('sharpe')} MaxDD={train_metrics.get('max_drawdown',0):.1%} IC={train_metrics.get('ic_mean',0):.4f}\n"
            + (f"样本外 Sharpe={test_metrics.get('sharpe')} MaxDD={test_metrics.get('max_drawdown',0):.1%}\n" if test_metrics else "")
            + f"结论: {'通过 ✓' if passed else '失败 ✗'}"
        )

        if passed:
            record_success(train_metrics, factor_cls.__name__, params)
            if oos_sharpe > self.best_sharpe:
                self.best_sharpe = oos_sharpe
            logger.info(f"[✓] {factor_name} PASSED | OOS Sharpe={oos_sharpe:.2f}")
            # 通知 Expert 和秘书：找到有效因子！
            _send_to_agent("agent:expert:main",
                f"[Alpha报告] 找到有效因子！{factor_name}\n{summary}\n请分析并更新 expert_guidance.md")
            _send_to_agent("agent:secretary:main",
                f"[重大进展] 第{self.iteration}轮迭代���到有效因子！{factor_name}\n{summary}\n请立即发邮件通知 YZX")
        else:
            reasons = []
            if train_metrics.get('sharpe', 0) <= 1.5:
                reasons.append(f"Sharpe={train_metrics['sharpe']:.2f}<1.5")
            if train_metrics.get('max_drawdown', -1) < -0.20:
                reasons.append(f"MaxDD={train_metrics['max_drawdown']:.1%}>20%")
            if train_metrics.get('ic_mean', 0) <= 0.05:
                reasons.append(f"IC={train_metrics['ic_mean']:.4f}<0.05")
            reason = " | ".join(reasons)
            record_failure(factor_name, reason, train_metrics, params)
            logger.info(f"[✗] {factor_name} FAILED | {reason}")
            # ���5轮通知 Expert 分析
            if self.iteration % 5 == 0:
                _send_to_agent("agent:expert:main",
                    f"[Alpha报告] 第{self.iteration}轮迭代，最近5轮均失败。\n最新结果：{summary}\n请更新优���建议到 expert_guidance.md")
            # 通知秘书记录
            _send_to_agent("agent:secretary:main",
                f"[迭代记录] 第{self.iteration}轮 {factor_name} {('通过' if passed else '失败')} | {reason if not passed else f'OOS Sharpe={oos_sharpe:.2f}'}")

        append_iteration_log(self.iteration, summary)
        lib = load_factor_library()
        failed = load_failed_factors()
        update_iteration_header(self.iteration, len(lib["factors"]), len(failed["factors"]))

        # 检测 Alpha 卡死：如果迭代时间异常，秘书会发现（通过日志时间戳）

    def _final_report(self):
        lib = load_factor_library()
        failed = load_failed_factors()
        logger.info(f"\n{'='*60}")
        logger.info(f"[OUROBOROS v3] 完成 {self.iteration} 次迭代")
        logger.info(f"有效因子: {len(lib['factors'])} 个")
        logger.info(f"失败: {len(failed['factors'])} 个")
        logger.info(f"最佳样本��� Sharpe: {self.best_sharpe:.2f}")
        if lib['factors']:
            for f in lib['factors']:
                m = f['metrics']
                logger.info(f"  {m['factor_name']}: Sharpe={m['sharpe']} MaxDD={m['max_drawdown']:.1%} IC={m['ic_mean']}")
        logger.info('='*60)
        # 最终汇报给秘书
        _send_to_agent("agent:secretary:main",
            f"[最终报告] OUROBOROS v3 完成 {self.iteration} 次迭代\n有效因子: {len(lib['factors'])} 个\n失败: {len(failed['factors'])} 个\n最佳OOS Sharpe: {self.best_sharpe:.2f}\n请发邮件汇报给 YZX")
