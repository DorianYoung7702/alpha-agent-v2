"""
v2_2_adx25.py - v2.2 ADX25 优化版
在 v2.0 基础上放宽 ADX 至 25（牛市时降至 15）
解决 2020 年达标问题，保持其他年份表现

核心变化：ADX_BASE = 25（原 35）
其他不变：EMA20/60、ATR0.8、���币30%、总仓150%、熊市套利
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
EMA_FAST = 20; EMA_SLOW = 60
ADX_BASE = 25   # 放宽至25，解决2020年问题
ATR_MULT = 0.8
MAX_SINGLE_WEIGHT = 0.30
MAX_TOTAL_WEIGHT  = 1.50

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
    try: return panel.xs(sym, level='symbol')['funding_rate'].shift(1).fillna(0)
    except: return pd.Series(dtype=float)

funding_series = {sym: get_funding_series(sym) for sym in SYMS}

def get_bear_return(date):
    r = [abs(funding_series[s].loc[date])*3
         if date in funding_series[s].index and abs(funding_series[s].loc[date]) >= FUNDING_THRESH
         else LENDING_YIELD for s in SYMS]
    return float(np.mean(r))

def make_ema_sig(sym):
    ohlcv = data[sym]
    c     = ohlcv['close']
    ema_f = c.ewm(span=EMA_FAST, adjust=False).mean()
    ema_s = c.ewm(span=EMA_SLOW, adjust=False).mean()
    ma200 = c.rolling(200).mean()
    adx   = calc_adx(ohlcv, window=14)
    bull_r  = bull_market.reindex(adx.index).fillna(False)
    adx_thr = pd.Series(float(ADX_BASE), index=adx.index)
    adx_thr[bull_r] = 15.0  # 牛市降至15
    sig = ((ema_f > ema_s) & (c > ma200) & (adx > adx_thr)).astype(int)
    sig = sig & btc_above_ma200.reindex(sig.index).fillna(False)
    return sig.shift(1).fillna(0).astype(int)

def run_sym(sym, base_w=0.2, start=None):
    sig    = make_ema_sig(sym)
    engine = TimeSeriesEngine(atr_mult=ATR_MULT, use_atr_stop=True)
    r      = engine.run(sig, data[sym], start_date=start)
    if not isinstance(r, pd.DataFrame) or r.empty:
        return pd.Series(dtype=float)
    rets = []
    cum_ret = 0.0
    for date, row in r.iterrows():
        is_bull = bull_market.loc[date]     if date in bull_market.index     else False
        btc_ok  = btc_above_ma200.loc[date] if date in btc_above_ma200.index else False
        w = min(base_w * (2.0 if is_bull else 1.0), MAX_SINGLE_WEIGHT)
        if not btc_ok:
            rets.append(get_bear_return(date) * base_w)
            cum_ret = 0.0
        elif row['position'] == 1:
            cum_ret += row['return']
            if cum_ret > 0.05: w = min(w*1.5, MAX_SINGLE_WEIGHT)
            rets.append(row['return'] * w)
        else:
            rets.append(STABLE_YIELD * base_w)
            cum_ret = 0.0
    return pd.Series(rets, index=r.index)

def run_portfolio(start='2020-01-01'):
    all_rets = [run_sym(s, 0.2, start) for s in SYMS]
    combined = pd.concat([r for r in all_rets if not r.empty], axis=1).fillna(0)
    scale = combined.abs().sum(axis=1).apply(
        lambda x: min(1.0, MAX_TOTAL_WEIGHT/x) if x > MAX_TOTAL_WEIGHT else 1.0)
    return combined.multiply(scale, axis=0).sum(axis=1)

def report_and_save(ret, label):
    m   = full_metrics(ret)
    oos = ret[ret.index >= '2022-01-01']
    mo  = full_metrics(oos)
    targets = {2020:0.60, 2021:1.50, 2022:0.0, 2023:0.80, 2024:0.60, 2025:0.15}
    print(f'\n[{label}]')
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
    print(f'\n  验收 vs v2.0基准:')
    print(f'    OOS Sharpe={mo["sharpe"]:.3f} 目标>1.77 {"✅" if mo["sharpe"]>=1.77 else "❌"}')
    print(f'    MaxDD={m["max_drawdown"]:.1%} 目标>-33% {"✅" if m["max_drawdown"]>-0.33 else "❌"}')
    print(f'    逐年达标={passed}/6 {"✅" if passed==6 else "❌"}')
    lines = [
        f'# {label}',
        f'\n**时间**: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}',
        f'\n## 核心指标',
        f'| 指标 | 全期 | OOS |',
        f'|------|------|-----|',
        f'| Sharpe | {m["sharpe"]:.3f} | {mo["sharpe"]:.3f} |',
        f'| MaxDD | {m["max_drawdown"]:.1%} | {mo["max_drawdown"]:.1%} |',
        f'| 年化 | {m["annual_return"]:.1%} | {mo["annual_return"]:.1%} |',
        f'\n## 验收',
        f'- OOS Sharpe={mo["sharpe"]:.3f} 目标>1.77 {"✅" if mo["sharpe"]>=1.77 else "❌"}',
        f'- MaxDD={m["max_drawdown"]:.1%} {"✅" if m["max_drawdown"]>-0.33 else "❌"}',
        f'- 逐年达标={passed}/6 {"✅" if passed==6 else "❌"}',
    ]
    Path(r'D:\YZX\shared\results\aggressive_latest.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'  [Saved] aggressive_latest.md')
    return m, mo

if __name__ == '__main__':
    print(f'Running v2.2 ADX{ADX_BASE} (放宽ADX提升2020年达标)...')
    ret = run_portfolio('2020-01-01')
    report_and_save(ret, f'v2.2 EMA{EMA_FAST}/{EMA_SLOW} ADX{ADX_BASE} ATR{ATR_MULT}')
    print(f'\nTotal: {time.time()-t0:.1f}s')
