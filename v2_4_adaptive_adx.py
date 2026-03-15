"""
v2_4_adaptive_adx.py - 自适应ADX阈值
在 v2.0 基础上用滚动中位数替代固定ADX阈值
adx_thresh_dynamic = (adx.rolling(60).median() * 1.2).clip(20, 50)
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
SYMS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','AVAXUSDT']
EMA_FAST = 20; EMA_SLOW = 60; ATR_MULT = 0.7
MAX_SINGLE_WEIGHT = 0.30; MAX_TOTAL_WEIGHT = 1.50

t0 = time.time()
data    = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMS, use_cache=True)
panel   = build_panel(data, funding_data=funding, min_history_days=60)

btc             = data['BTCUSDT']
btc_close       = btc['close']
btc_ma200       = btc_close.rolling(200).mean()
btc_above_ma200 = (btc_close > btc_ma200).shift(1).fillna(False)
btc_adx_global  = calc_adx(btc, window=20).shift(1).fillna(0)
above_10d       = btc_above_ma200.rolling(10).min().fillna(0).astype(bool)
bull_market     = (above_10d & (btc_adx_global > 25)).shift(1).fillna(False)

def get_funding_series(sym):
    try: return panel.xs(sym,level='symbol')['funding_rate'].shift(1).fillna(0)
    except: return pd.Series(dtype=float)
funding_series = {sym: get_funding_series(sym) for sym in SYMS}

def get_bear_return(date):
    r = [abs(funding_series[s].loc[date])*3
         if date in funding_series[s].index and abs(funding_series[s].loc[date]) >= FUNDING_THRESH
         else LENDING_YIELD for s in SYMS]
    return float(np.mean(r))

def make_adaptive_sig(sym):
    ohlcv = data[sym]
    c     = ohlcv['close']
    ema_f = c.ewm(span=EMA_FAST, adjust=False).mean()
    ema_s = c.ewm(span=EMA_SLOW, adjust=False).mean()
    ma200 = c.rolling(200).mean()
    adx   = calc_adx(ohlcv, window=14)
    # 自适应ADX阈值
    adx_median_60  = adx.rolling(60).median()
    adx_thresh_dyn = (adx_median_60 * 1.2).clip(20, 50)
    # 牛市时阈值再降10
    bull_r = bull_market.reindex(adx.index).fillna(False)
    adx_thresh_dyn[bull_r] = (adx_thresh_dyn[bull_r] - 10).clip(15, 50)
    sig = ((ema_f > ema_s) & (c > ma200) & (adx > adx_thresh_dyn)).astype(int)
    sig = sig & btc_above_ma200.reindex(sig.index).fillna(False)
    return sig.shift(1).fillna(0).astype(int)

def run_sym(sym, base_w=0.2, start=None):
    sig    = make_adaptive_sig(sym)
    engine = TimeSeriesEngine(atr_mult=ATR_MULT, use_atr_stop=True)
    r      = engine.run(sig, data[sym], start_date=start)
    if not isinstance(r, pd.DataFrame) or r.empty: return pd.Series(dtype=float)
    rets = []; cum_ret = 0.0
    for date, row in r.iterrows():
        is_bull = bull_market.loc[date]     if date in bull_market.index     else False
        btc_ok  = btc_above_ma200.loc[date] if date in btc_above_ma200.index else False
        w = min(base_w * (2.0 if is_bull else 1.0), MAX_SINGLE_WEIGHT)
        if not btc_ok:
            rets.append(get_bear_return(date) * base_w); cum_ret = 0.0
        elif row['position'] == 1:
            cum_ret += row['return']
            if cum_ret > 0.05: w = min(w*1.5, MAX_SINGLE_WEIGHT)
            rets.append(row['return'] * w)
        else:
            rets.append(STABLE_YIELD * base_w); cum_ret = 0.0
    return pd.Series(rets, index=r.index)

def run_portfolio(start='2020-01-01'):
    all_rets = [run_sym(s, 0.2, start) for s in SYMS]
    combined = pd.concat([r for r in all_rets if not r.empty], axis=1).fillna(0)
    scale = combined.abs().sum(axis=1).apply(
        lambda x: min(1.0, MAX_TOTAL_WEIGHT/x) if x > MAX_TOTAL_WEIGHT else 1.0)
    return combined.multiply(scale, axis=0).sum(axis=1)

targets = {2020:0.60,2021:1.50,2022:0.0,2023:0.80,2024:0.60,2025:0.15}
ret = run_portfolio('2020-01-01')
m   = full_metrics(ret)
oos = ret[ret.index >= '2022-01-01']
mo  = full_metrics(oos)
passed = 0
print('\n[v2.4 自适应ADX EMA20/60 ATR0.7]')
print('  全期: Sh=%.3f DD=%.1f%% Ann=%.1f%%'%(m['sharpe'],m['max_drawdown']*100,m['annual_return']*100))
print('  OOS:  Sh=%.3f DD=%.1f%% Ann=%.1f%%'%(mo['sharpe'],mo['max_drawdown']*100,mo['annual_return']*100))
for yr in range(2020,2026):
    r=ret[ret.index.year==yr]
    if len(r)>5:
        mm=full_metrics(r); t=targets.get(yr,0)
        f='✓' if mm['annual_return']>=t else '✗'
        if mm['annual_return']>=t: passed+=1
        print('    %d: Ann=%+.1f%% Sh=%+.3f DD=%.1f%% %s'%(
            yr,mm['annual_return']*100,mm['sharpe'],mm['max_drawdown']*100,f))
print('  逐年达标: %d/6'%passed)
print('  验收: OOS Sh=%.3f %s MaxDD=%.1f%% %s ���标=%d/6 %s'%(
    mo['sharpe'], '✅' if mo['sharpe']>=1.75 else '❌',
    m['max_drawdown']*100, '✅' if m['max_drawdown']>-0.33 else '❌',
    passed, '✅' if passed==6 else '❌'))
lines=[
    '# v2.4 自适应ADX 回测结果',
    '\n**时间**: '+pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
    '\n## 核心指标',
    '| 指标 | 全期 | OOS |','|------|------|-----|',
    '| Sharpe | %.3f | %.3f |'%(m['sharpe'],mo['sharpe']),
    '| MaxDD | %.1f%% | %.1f%% |'%(m['max_drawdown']*100,mo['max_drawdown']*100),
    '| 年化 | %.1f%% | %.1f%% |'%(m['annual_return']*100,mo['annual_return']*100),
    '\n## 验收',
    '- OOS Sh=%.3f 目标>1.75 %s'%(mo['sharpe'],'✅' if mo['sharpe']>=1.75 else '❌'),
    '- MaxDD=%.1f%% %s'%(m['max_drawdown']*100,'✅' if m['max_drawdown']>-0.33 else '❌'),
    '- 达标=%d/6 %s'%(passed,'✅' if passed==6 else '❌'),
]
Path(r'D:\YZX\shared\results\aggressive_latest.md').write_text('\n'.join(lines),encoding='utf-8')
print('  [Saved] aggressive_latest.md')
print('Total: %.1fs'%(time.time()-t0))
