"""
v3_0_dynamic.py - 动态选币策略
扩展到市值前15币，用截面动量动态轮换选最强5-8币
基于 aggressive_v4.py 修改
"""
import sys
sys.path.insert(0, r'D:\YZX\alpha-agent\src')
from loguru import logger; logger.remove(); logger.add(sys.stdout, level='WARNING')
import pandas as pd
import numpy as np
from data.fetcher import fetch_all_symbols, fetch_funding_all, build_panel, SYMBOLS_TOP30
from factors.ts_signals_v2 import calc_adx
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics
from pathlib import Path
import time

STABLE_YIELD   = 0.05/365
LENDING_YIELD  = 0.08/365
FUNDING_THRESH = 0.00005
# 扩展到前15个市值币
SYMS_POOL = [
    'BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT',
    'ADAUSDT','AVAXUSDT','DOTUSDT','LINKUSDT','LTCUSDT',
    'ATOMUSDT','UNIUSDT','NEARUSDT','INJUSDT','APTUSDT'
]
TOP_N = 8   # 每日选最强8个币
EMA_FAST = 20; EMA_SLOW = 60; ADX_BASE = 35; ATR_MULT = 0.8
MAX_SINGLE_WEIGHT = 0.20  # 扩展到15币后单币上限降至20%

t0 = time.time()
print('Loading data...')
data    = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMS_POOL, use_cache=True)
panel   = build_panel(data, funding_data=funding, min_history_days=60)
prices  = panel['close'].unstack(level='symbol')

btc             = data['BTCUSDT']
btc_close       = btc['close']
btc_ma200       = btc_close.rolling(200).mean()
btc_above_ma200 = (btc_close > btc_ma200).shift(1).fillna(False)
btc_adx_global  = calc_adx(btc, window=20).shift(1).fillna(0)
above_10d       = btc_above_ma200.rolling(10).min().fillna(0).astype(bool)
bull_market     = (above_10d & (btc_adx_global > 25)).shift(1).fillna(False)

def get_funding_series(sym):
    try: return panel.xs(sym, level='symbol')['funding_rate'].shift(1).fillna(0)
    except: return pd.Series(dtype=float)

funding_series = {sym: get_funding_series(sym) for sym in SYMS_POOL}

def get_bear_return(date):
    r = [abs(funding_series[s].loc[date])*3
         if date in funding_series[s].index and abs(funding_series[s].loc[date]) >= FUNDING_THRESH
         else LENDING_YIELD for s in SYMS_POOL[:5]]
    return float(np.mean(r))

# ���─ 截面动量选币（20日收益率排名）──────────────────────────
def compute_momentum_rank(date, lookback=20):
    """计算date当日各币的20日动量排名，返回最强TOP_N"""
    try:
        date_idx = prices.index.get_loc(date)
        if date_idx < lookback:
            return None
        # 用前一天的数据（防未来）
        past_date = prices.index[date_idx - lookback]
        prev_date = prices.index[date_idx - 1]
        rets = {}
        for sym in SYMS_POOL:
            if sym in prices.columns:
                p_now  = prices.loc[prev_date, sym] if prev_date in prices.index else np.nan
                p_past = prices.loc[past_date, sym] if past_date in prices.index else np.nan
                if not np.isnan(p_now) and not np.isnan(p_past) and p_past > 0:
                    rets[sym] = (p_now - p_past) / p_past
        if len(rets) < TOP_N:
            return None
        sorted_syms = sorted(rets, key=rets.get, reverse=True)
        return sorted_syms[:TOP_N]
    except Exception:
        return None

# ── 各币EMA信号 ──────────────────────────────────────────
def make_ema_sig(sym):
    ohlcv = data[sym]
    c     = ohlcv['close']
    ema_f = c.ewm(span=EMA_FAST, adjust=False).mean()
    ema_s = c.ewm(span=EMA_SLOW, adjust=False).mean()
    ma200 = c.rolling(200).mean()
    adx   = calc_adx(ohlcv, window=14)
    bull_r  = bull_market.reindex(adx.index).fillna(False)
    adx_thr = pd.Series(float(ADX_BASE), index=adx.index)
    adx_thr[bull_r] = 25.0
    sig = ((ema_f > ema_s) & (c > ma200) & (adx > adx_thr)).astype(int)
    sig = sig & btc_above_ma200.reindex(sig.index).fillna(False)
    return sig.shift(1).fillna(0).astype(int)

print('Computing signals...')
signals = {sym: make_ema_sig(sym) for sym in SYMS_POOL if sym in data}
engines = {sym: TimeSeriesEngine(atr_mult=ATR_MULT, use_atr_stop=True) for sym in signals}
results = {}
for sym, sig in signals.items():
    r = engines[sym].run(sig, data[sym], start_date='2020-01-01')
    if isinstance(r, pd.DataFrame) and not r.empty:
        results[sym] = r

# ── 组合回测（动态选币）──────────────────────���───────────
print('Running portfolio...')
all_dates = sorted(set.intersection(*[set(r.index) for r in results.values()]))
daily_portfolio = []

for date in all_dates:
    btc_ok  = btc_above_ma200.loc[date] if date in btc_above_ma200.index else False
    is_bull = bull_market.loc[date]     if date in bull_market.index     else False

    if not btc_ok:
        # 熊市：套利
        daily_portfolio.append(get_bear_return(date) * 0.2)
        continue

    # 动态选���：截面动量TOP_N + EMA信号过滤
    top_syms = compute_momentum_rank(date)
    if top_syms is None:
        daily_portfolio.append(STABLE_YIELD * 0.2)
        continue

    # 在TOP_N中筛选有趋势信号的币
    active_syms = [s for s in top_syms
                   if s in results and date in results[s].index
                   and results[s].loc[date, 'position'] == 1]

    if not active_syms:
        daily_portfolio.append(STABLE_YIELD * 0.2)
        continue

    # 等权分配
    w = min(MAX_SINGLE_WEIGHT, 1.0 / len(active_syms))
    if is_bull: w = min(w * 1.5, MAX_SINGLE_WEIGHT)
    day_ret = sum(results[s].loc[date, 'return'] * w for s in active_syms)
    daily_portfolio.append(day_ret)

ret = pd.Series(daily_portfolio, index=all_dates)

# ── 报告 ────────────────────────────���─────────────────────
m   = full_metrics(ret)
oos = ret[ret.index >= '2022-01-01']
mo  = full_metrics(oos)
targets = {2020:0.60, 2021:1.50, 2022:0.0, 2023:0.80, 2024:0.60, 2025:0.15}
print(f'\n[v3.0 动态选币（TOP{TOP_N}，前15币池）]')
print(f'  全期: Sh={m["sharpe"]:.3f} DD={m["max_drawdown"]:.1%} Ann={m["annual_return"]:.1%}')
print(f'  OOS:  Sh={mo["sharpe"]:.3f} DD={mo["max_drawdown"]:.1%} Ann={mo["annual_return"]:.1%}')
passed = 0
for yr in range(2020, 2026):
    r = ret[ret.index.year == yr]
    if len(r) > 5:
        mm = full_metrics(r)
        t  = targets.get(yr, 0)
        f  = '✓' if mm['annual_return'] >= t else '✗'
        if mm['annual_return'] >= t: passed += 1
        print(f'    {yr}: Ann={mm["annual_return"]:+.1%} Sh={mm["sharpe"]:+.3f} DD={mm["max_drawdown"]:.1%} {f}')
print(f'  逐年达标: {passed}/6')

# 保存结果
lines = [
    f'# v3.0 动态选币 回测结果',
    f'\n**时间**: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}',
    f'\n## 核心指标',
    f'| 指标 | 全期 | OOS |',
    f'|------|------|-----|',
    f'| Sharpe | {m["sharpe"]:.3f} | {mo["sharpe"]:.3f} |',
    f'| MaxDD | {m["max_drawdown"]:.1%} | {mo["max_drawdown"]:.1%} |',
    f'| 年化 | {m["annual_return"]:.1%} | {mo["annual_return"]:.1%} |',
    f'\n## 参数',
    f'- 币池：前15市值币',
    f'- 动态选币：20日动量TOP{TOP_N} + EMA{EMA_FAST}/{EMA_SLOW}信号过滤',
    f'- ADX{ADX_BASE} ATR{ATR_MULT} 单币上限{MAX_SINGLE_WEIGHT*100:.0f}%',
    f'\n## 逐年达标：{passed}/6',
]
Path(r'D:\YZX\shared\results\aggressive_latest.md').write_text('\n'.join(lines), encoding='utf-8')
print(f'\n[Saved] aggressive_latest.md')
print(f'Total: {time.time()-t0:.1f}s')
