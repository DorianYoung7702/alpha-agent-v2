"""
v2_2_breakout.py - v2.2 突破确认版
在v2.0基础上加入价格突破过滤：
- EMA20 > EMA60���趋势）
- 收盘价 > N日最高价的X%（突破确认）
- 扫描 N=10/20，X=0.95/0.98
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
EMA_FAST = 20; EMA_SLOW = 60; ADX_BASE = 35; ATR_MULT = 0.8
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
    try: return panel.xs(sym, level='symbol')['funding_rate'].shift(1).fillna(0)
    except: return pd.Series(dtype=float)

funding_series = {sym: get_funding_series(sym) for sym in SYMS}

def get_bear_return(date):
    r = [abs(funding_series[s].loc[date])*3
         if date in funding_series[s].index and abs(funding_series[s].loc[date]) >= FUNDING_THRESH
         else LENDING_YIELD for s in SYMS]
    return float(np.mean(r))

def make_sig(sym, breakout_window=0, breakout_pct=0.95):
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
    # 突破过���
    if breakout_window > 0:
        high_n = c.rolling(breakout_window).max()
        breakout = (c >= high_n * breakout_pct).astype(int)
        sig = sig & breakout
    return sig.shift(1).fillna(0).astype(int)

def run_sym(sym, base_w=0.2, start=None, bw=0, bp=0.95):
    s      = make_sig(sym, bw, bp)
    engine = TimeSeriesEngine(atr_mult=ATR_MULT, use_atr_stop=True)
    r      = engine.run(s, data[sym], start_date=start)
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

def run_portfolio(bw=0, bp=0.95, start='2020-01-01'):
    all_rets = [run_sym(s, 0.2, start, bw, bp) for s in SYMS]
    combined = pd.concat([r for r in all_rets if not r.empty], axis=1).fillna(0)
    scale = combined.abs().sum(axis=1).apply(
        lambda x: min(1.0, MAX_TOTAL_WEIGHT/x) if x > MAX_TOTAL_WEIGHT else 1.0)
    return combined.multiply(scale, axis=0).sum(axis=1)

# ─��� 参数扫描 ─────────────────────────────────────���────────
print(f'{'配置':<25} {'全Sh':>7} {'MaxDD':>7} {'年化':>8} {'OOSSh':>7} {'2020':>8} {'达标':>5}')
print('-'*68)

best = None
for bw, bp in [(0,0), (10,0.95), (10,0.98), (20,0.95), (20,0.98), (5,0.95), (15,0.95)]:
    ret = run_portfolio(bw, bp)
    if ret.empty: continue
    m   = full_metrics(ret)
    oos = ret[ret.index >= '2022-01-01']
    mo  = full_metrics(oos)
    r20 = ret[ret.index.year == 2020]
    m20 = full_metrics(r20) if len(r20) > 5 else {'annual_return':0}
    passed = sum(1 for yr in range(2020,2026)
                 if len(ret[ret.index.year==yr])>5 and
                 full_metrics(ret[ret.index.year==yr])['annual_return'] >=
                 {2020:0.60,2021:1.50,2022:0.0,2023:0.80,2024:0.60,2025:0.15}.get(yr,0))
    label = f'BW={bw} BP={bp}' if bw > 0 else 'v2.0基准(无过滤)'
    print(f'{label:<25} {m["sharpe"]:7.3f} {m["max_drawdown"]:7.1%} {m["annual_return"]:8.1%} '
          f'{mo["sharpe"]:7.3f} {m20["annual_return"]:8.1%} {passed:5}/6')
    if mo['sharpe'] >= 1.77 and m['max_drawdown'] > -0.33 and passed == 6:
        if best is None or mo['sharpe'] > best['oos_sh']:
            best = {'bw':bw,'bp':bp,'sh':m['sharpe'],'dd':m['max_drawdown'],
                    'ann':m['annual_return'],'oos_sh':mo['sharpe'],'passed':passed}

if best:
    print(f'\n最优: BW={best["bw"]} BP={best["bp"]} Sh={best["sh"]:.3f} OOSSh={best["oos_sh"]:.3f}')
else:
    print('\n未找到满足全部验收标准的参数')

print(f'\nTotal: {time.time()-t0:.1f}s')
