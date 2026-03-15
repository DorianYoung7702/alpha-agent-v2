# OUROBOROS Alpha-Agent-Aggressive

> v1.4 激进版量化策略 | DynamicPosition + BullMultiplier | PRODUCTION

## 项目简介

OUROBOROS 是一个全自动量化因子挖掘闭环系统，由 AI Agent 替代量化研究员执行���究迭代。

本仓库是**激进版策略**，目标用户为风险偏好较高的投资者。稳健版见 [alpha-agent](https://github.com/DorianYoung7702/alpha-agent)。

### 核心指标

| 指标 | 全期 | OOS（样本外）|
|------|------|------|
| Sharpe | **1.905** | **1.629** |
| 最大回撤 | -26.0% | -25.2% |
| 年化收益 | +99% | +62% |
| fee=0.05% OOS | — | 1.560 |
| 2022熊市持仓 | **0天** | ✅ |
| 逐年达标 | **6/6** | ✅ |

Expert 正式批准：2026-03-15

---

## 项目架构

```
alpha-agent-aggressive/
│
├── aggressive_v4.py          # ★ 核心策略（PRODUCTION，禁止修改）
├── aggressive_live.py        # ★ OKX 实盘接入（干跑/真实下单）
│
├── src/                      # 基础库（与稳健版共享）
│   ├── backtest/
│   │   ├── engine_ts.py      # ���序回测引擎（ATR止损）
│   │   ├── engine.py         # 截面回测引擎
│   │   └── metrics.py        # 绩效指标计算
│   ├── data/
│   │   └── fetcher.py        # Binance数据拉取（含资金费率）
│   ├── factors/
│   │   ├── ts_signals_v2.py  # ADX / MACD / MA 信号
│   │   ├── volatility.py     # 历史波动率因子
│   │   └── ...               # 其他因子
│   └── utils/
│       ���── memory.py         # Agent 记忆工具
│
├── data/raw/                 # 行情缓存（parquet，30个币种）
│
├── reports/
│   └── strategy_report.png  # 回测可视化报告
│
├── logs/
│   ├─��� live_YYYY-MM-DD.log  # 实盘运行日志
│   └── live_trades.json     # 交易记录（干跑/实盘）
│
├── memory/                   # Agent 记忆（本地，不提交）
│
├── v1_4_bear.py              # 熊市补位模块（开发中）
│
└── [辅助研究脚本]
    ├── grid_search*.py       # 参数网格搜索
    ├── walk_forward.py       # Walk-forward 验证
    ├── visualize.py          # 可视化
    ├── overfit_test.py       # 过拟合检验
    └── eval_*.py             # 各类评估脚本
```

---

## 策略逻辑

### 信号架���

```
全局门控：BTC close > BTC MA200（shift 1，防未来函数）
    │
    ├─ 牛市确认：BTC > MA200 连续10天 + ADX(20) > 25
    │   └─ 仓位倍数 ×2.0（单币最大 40%）
    │
    ├─ 趋势���号：MACD(12,26,9) + MA200 + ADX > 35（牛市降至20）
    │   └─ ATR × 1.0 动态止损
    │
    ├─ 动量加仓：持仓盈利 > 5% 时加 0.5x
    │
    └─ BB均值回归：震荡市 ADX < 25，BB(10, 1.5) ���控

BTC < MA200（熊市）：全部空仓
```

### 5个标的

| 币种 | 基础权重 | 牛市权重 |
|------|---------|----------|
| BTCUSDT | 20% | 40% |
| ETHUSDT | 20% | 40% |
| SOLUSDT | 20% | 40% |
| BNBUSDT | 20% | 40% |
| AVAXUSDT | 20% | 40% |

---

## 快速开始

### 环境要求

```bash
pip install pandas numpy loguru requests
# Python: C:\Users\Administrator\Miniconda3\python.exe
```

### 1. 运行回测

```bash
# 完整回测（2020-2026）
python aggressive_v4.py
```

输出示例：
```
全期: Sh=1.905 DD=-26.0% Ann=+99%
OOS:  Sh=1.629 DD=-25.2% Ann=+62%
2022: Ann=+3.5% DD=0.0% ✓（完全空仓）
```

### 2. 生成可视化报告

```bash
python D:\YZX\shared\generate_reports.py aggressive
# 输出：reports/strategy_report.png
```

### 3. 干跑实盘（不下单）

```bash
# 确认 DRY_RUN = True（默认）
python aggressive_live.py
```

输出示例：
```
17:06:38 | INFO | BTC-USDT: sig=0 bull=False btc_ok=False weight=0%
17:06:38 | INFO | 当前状态：BTC < MA200，全部空仓
```

### 4. 开启真实下单

编辑 `aggressive_live.py` 第17行：
```python
DRY_RUN = False  # 改为 False
```

⚠️ **开启前请确认：**
- API Key 权限：只读 + 交易（禁止提币）
- 资金规模已确认
- 干跑信号已观察至少3天

---

## 定时任务

系统已配置两个每日 cron：

| 任务 | 时间（UTC）| 说明 |
|------|-----------|------|
| trading-daily-run | 00:05 | 运行实盘脚本，汇报给毛毛 |
| aggressive-v14-dryrun | 00:10 | Alpha 干跑检查信号 |

---

## 回���验证（9/9 通过）

| 验证项 | 结果 | 状态 |
|--------|------|------|
| OOS测试 | Sharpe=1.629 | ✅ |
| 参���敏感性 | CV=1.1% | ✅ |
| 费率压力 | fee=0.10% OOS=1.543 | ✅ |
| 2022熊市 | 完全空仓（0天）| ✅ |
| 杠杆评估 | 1.0x最优 | ✅ |
| 逐年达标 | 6/6年 | ✅ |
| 换标的（主流）| OOS=1.296 | ✅ |
| 换标的（随���）| 均值=1.196 | ✅ |
| Walk-Forward | CV=0%, Sh=1.839 | ✅ |

---

## 相关仓库

| 仓库 | 说明 |
|------|------|
| [alpha-agent](https://github.com/DorianYoung7702/alpha-agent) | 稳健版 v3.2，Sharpe=2.035，MaxDD=-11.9% |
| [alpha-agent-aggressive](https://github.com/DorianYoung7702/alpha-agent-aggressive) | 本仓库，激进版 v1.4 |

---

## 注���事项

- **禁止修改** `aggressive_v4.py`（PRODUCTION 版本，Expert 已批准）
- **禁止使用未来函数**：���有信号强制 `shift(1)`
- **API 密钥**：存储在 `D:\YZX\shared\okx_config.py`，已加入 `.gitignore`，禁止提���
- `data/raw/` 为行情缓存，体积较大，已纳入 git 追踪
- `memory/` 为 Agent 本地记忆，不提交

---

*OUROBOROS | Alpha Agent | 2026-03-15*
