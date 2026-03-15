# alpha-agent-aggressive

OUROBOROS 激进版策略 v1.4

## 策略概述

- **版本**: v1.4 Aggressive (DynamicPosition + BullMultiplier)
- **标的**: BTCUSDT / ETHUSDT / SOLUSDT / BNBUSDT / AVAXUSDT
- **状态**: PRODUCTION（Expert 正式批准 2026-03-15）

## 核心参数

| 参数 | 值 |
|------|----|
| 基础权重 | 5币等权 20% |
| ADX阈��� | 35（牛市放宽至20）|
| 牛市确认 | BTC MA200 + ADX>25 连续10天 |
| 牛市仓位倍数 | 2.0x（单币上���40%）|
| 动量加仓 | 持仓盈利>5% 加0.5x |
| ATR止损 | 1.0x |
| BB均值回归 | 震荡市 ADX<25，MA200门控 |
| 全局门控 | BTC close > BTC MA200 (shift 1) |

## 回测结果

| 指标 | 全期 | OOS |
|------|------|-----|
| Sharpe | 1.905 | 1.629 |
| 最大回撤 | -26.0% | -25.2% |
| 年化收益 | +99% | +62% |
| fee=0.05% OOS Sharpe | — | 1.560 |

## 逐年达标（6/6）

| 年份 | 年化 | 目标 |
|------|------|------|
| 2020 | +63.5% | >60% ✅ |
| 2021 | +464% | >150% ✅ |
| 2022 | +3.5% | >0% ✅ (完全空仓) |
| 2023 | +199% | >80% ✅ |
| 2024 | +110% | >60% ✅ |
| 2025 | +16.8% | >15% ✅ |

## 依赖

```
pip install pandas numpy loguru
```

策略共享 `D:\VC\alpha-agent\src` 作为基础库路径。

## 运行

```bash
python aggressive_v4.py
```
