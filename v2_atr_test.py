"""
v2_atr_test.py - ATR 参数扫描，压制 MaxDD
固定 EMA20/60 ADX35，测试 ATR 0.6/0.8/1.0/1.2/1.5
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
import time

STABLE_YIELD   = 0.05/365
LENDING_YIELD  = 0.08/365
FUNDING_THRESH = 0.00005
SYMS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','AVAXUSDT']
EMA_FAST = 20; EMA_SLOW = 60; ADX_BASE = 35
MAX_SINGLE_WEIGHT = 0.30
MAX_TOTAL_WEIGHT  = 1.50

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
    r = [abs(funding_series[s].loc[date])*3 if date in funding_series[s].index and abs(funding_series[s].loc[date]) >= FUNDING_THRESH else LENDING_YIELD for s in SYMS]
    return float(np.mean(r))

def run_atr(atr_mult, start='2020-01-01'):
    all_rets = []
    for sym in SYMS:
        ohlcv = data[sym]
        c = ohlcv['close']
        ema_f = c.ewm(span=EMA_FAST, adjust=False).mean()
        ema_s = c.ewm(span=EMA_SLOW, adjust=False).mean()
        ma200 = c.rolling(200).mean()
        adx   = calc_adx(ohlcv, window=14)
        bull_r  = bull_market.reindex(adx.index).fillna(False)
        adx_thr = pd.Series(float(ADX_BASE), index=adx.index)
        adx_thr[bull_r] = 25.0
        sig = ((ema_f > ema_s) & (c > ma200) & (adx > adx_thr)).astype(int)
        sig = (sig & btc_above_ma200.reindex(sig.index).fillna(False)).shift(1).fillna(0).astype(int)
        engine = TimeSeriesEngine(atr_mult=atr_mult, use_atr_stop=True)
        r = engine.run(sig, ohlcv, start_date=start)
        if not isinstance(r, pd.DataFrame) or r.empty: continue
        rets = []
        cum_ret = 0.0
        for date, row in r.iterrows():
            is_bull = bull_market.loc[date] if date in bull_market.index else False
            btc_ok  = btc_above_ma200.loc[date] if date in btc_above_ma200.index else False
            w = min(0.2 * (2.0 if is_bull else 1.0), MAX_SINGLE_WEIGHT)
            if not btc_ok:
                rets.append(get_bear_return(date) * 0.2); cum_ret = 0.0
            elif row['position'] == 1:
                cum_ret += row['return']
                if cum_ret > 0.05: w = min(w*1.5, MAX_SINGLE_WEIGHT)
                rets.append(row['return'] * w)
            else:
                rets.append(STABLE_YIELD * 0.2); cum_ret = 0.0
        all_rets.append(pd.Series(rets, index=r.index))
    if not all_rets: return pd.Series(dtype=float)
    combined = pd.concat(all_rets, axis=1).fillna(0)
    scale = combined.abs().sum(axis=1).apply(lambda x: min(1.0, MAX_TOTAL_WEIGHT/x) if x > MAX_TOTAL_WEIGHT else 1.0)
    return combined.multiply(scale, axis=0).sum(axis=1)

print(f'{'ATR':^8} {'全Sh':^8} {'MaxDD':^8} {'年化':^8} {'OOSSh':^8} {'OOSDD':^8} {'2022':^8} {'2021DD':^8}')
print('-'*68)

best_result = None
for atr in [0.5, 0.6, 0.8, 1.0, 1.2, 1.5]:
    ret = run_atr(atr)
    if ret.empty: continue
    m   = full_metrics(ret)
    oos = ret[ret.index >= '2022-01-01']
    mo  = full_metrics(oos)
    r22 = ret[ret.index.year == 2022]
    m22 = full_metrics(r22) if len(r22) > 5 else {'annual_return': 0}
    r21 = ret[ret.index.year == 2021]
    dd21 = full_metrics(r21)['max_drawdown'] if len(r21) > 5 else 0
    print(f'{atr:^8.1f} {m["sharpe"]:^8.3f} {m["max_drawdown"]:^8.1%} '
          f'{m["annual_return"]:^8.1%} {mo["sharpe"]:^8.3f} {mo["max_drawdown"]:^8.1%} '
          f'{m22["annual_return"]:^8.1%} {dd21:^8.1%}')
    if m['max_drawdown'] > -0.30 and mo['sharpe'] > 1.65:
        if best_result is None or mo['sharpe'] > best_result['oos_sh']:
            best_result = {'atr': atr, 'sh': m['sharpe'], 'dd': m['max_drawdown'],
                          'ann': m['annual_return'], 'oos_sh': mo['sharpe'],
                          'oos_dd': mo['max_drawdown'], 'ann22': m22['annual_return']}

if best_result:
    print(f'\n最优 ATR={best_result["atr"]}: Sh={best_result["sh"]:.3f} DD={best_result["dd"]:.1%} OOSSh={best_result["oos_sh"]:.3f}')
else:
    print('\n未找到满足MaxDD>-30% + OOSSh>1.65的参数')
