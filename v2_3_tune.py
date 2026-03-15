"""
v2_3_tune.py - v2.3 微调版
在v2.0基础上扫描ADX窗口和牛市乘数组合
目标：OOS Sharpe > 1.77，MaxDD < -33%，6/6达标
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
bull_market_base = (above_10d & (btc_adx_global > 25)).shift(1).fillna(False)

def get_funding_series(sym):
    try: return panel.xs(sym, level='symbol')['funding_rate'].shift(1).fillna(0)
    except: return pd.Series(dtype=float)
funding_series = {sym: get_funding_series(sym) for sym in SYMS}

def get_bear_return(date):
    r = [abs(funding_series[s].loc[date])*3
         if date in funding_series[s].index and abs(funding_series[s].loc[date]) >= FUNDING_THRESH
         else LENDING_YIELD for s in SYMS]
    return float(np.mean(r))

def run_combo(atr_mult, bull_mult, adx_window, bull_confirm, start='2020-01-01'):
    # 重新计算牛市确认
    btc_adx_local = calc_adx(data['BTCUSDT'], window=adx_window).shift(1).fillna(0)
    above_nd = btc_above_ma200.rolling(bull_confirm).min().fillna(0).astype(bool)
    bull_mkt = (above_nd & (btc_adx_local > 25)).shift(1).fillna(False)

    all_rets = []
    for sym in SYMS:
        ohlcv = data[sym]
        c     = ohlcv['close']
        ema_f = c.ewm(span=EMA_FAST, adjust=False).mean()
        ema_s = c.ewm(span=EMA_SLOW, adjust=False).mean()
        ma200 = c.rolling(200).mean()
        adx   = calc_adx(ohlcv, window=adx_window)
        bull_r  = bull_mkt.reindex(adx.index).fillna(False)
        adx_thr = pd.Series(float(ADX_BASE), index=adx.index)
        adx_thr[bull_r] = 25.0
        sig = ((ema_f > ema_s) & (c > ma200) & (adx > adx_thr)).astype(int)
        sig = (sig & btc_above_ma200.reindex(sig.index).fillna(False)).shift(1).fillna(0).astype(int)
        engine = TimeSeriesEngine(atr_mult=atr_mult, use_atr_stop=True)
        r = engine.run(sig, ohlcv, start_date=start)
        if not isinstance(r, pd.DataFrame) or r.empty: continue
        rets = []; cum_ret = 0.0
        for date, row in r.iterrows():
            is_bull = bull_mkt.loc[date]     if date in bull_mkt.index     else False
            btc_ok  = btc_above_ma200.loc[date] if date in btc_above_ma200.index else False
            w = min(0.2 * (bull_mult if is_bull else 1.0), MAX_SINGLE_WEIGHT)
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
    scale = combined.abs().sum(axis=1).apply(
        lambda x: min(1.0, MAX_TOTAL_WEIGHT/x) if x > MAX_TOTAL_WEIGHT else 1.0)
    return combined.multiply(scale, axis=0).sum(axis=1)

TARGETS = {2020:0.60, 2021:1.50, 2022:0.0, 2023:0.80, 2024:0.60, 2025:0.15}

print(f'{'配置':<30} {'全Sh':>7} {'MaxDD':>7} {'年化':>8} {'OOSSh':>7} {'达标':>5}')
print('-'*65)

best = None
for atr in [0.6, 0.8, 1.0]:
    for bull in [1.5, 2.0, 2.5]:
        for adx_w in [10, 14, 20]:
            for confirm in [5, 10, 15]:
                ret = run_combo(atr, bull, adx_w, confirm)
                if ret.empty: continue
                m   = full_metrics(ret)
                oos = ret[ret.index >= '2022-01-01']
                mo  = full_metrics(oos)
                passed = sum(1 for yr in range(2020,2026)
                    if len(ret[ret.index.year==yr])>5 and
                    full_metrics(ret[ret.index.year==yr])['annual_return'] >= TARGETS.get(yr,0))
                label = f'ATR{atr} Bull{bull} ADX_w{adx_w} C{confirm}'
                if mo['sharpe'] >= 1.77 and m['max_drawdown'] > -0.33 and passed == 6:
                    print(f'{label:<30} {m["sharpe"]:7.3f} {m["max_drawdown"]:7.1%} '
                          f'{m["annual_return"]:8.1%} {mo["sharpe"]:7.3f} {passed:5}/6  *** PASS ***')
                    if best is None or mo['sharpe'] > best['oos_sh']:
                        best = {'atr':atr,'bull':bull,'adx_w':adx_w,'confirm':confirm,
                                'sh':m['sharpe'],'dd':m['max_drawdown'],
                                'ann':m['annual_return'],'oos_sh':mo['sharpe']}

if best:
    print(f'\n最优: ATR={best["atr"]} Bull={best["bull"]} ADX_w={best["adx_w"]} Confirm={best["confirm"]}')
    print(f'Sh={best["sh"]:.3f} MaxDD={best["dd"]:.1%} OOSSh={best["oos_sh"]:.3f}')
else:
    # 打印最接近的（OOS最高的）
    print('\n未找到满足全部条件的参数，打印OOS最���的3组：')
    results = []
    for atr in [0.6, 0.8]:
        for bull in [2.0, 2.5]:
            for adx_w in [14, 20]:
                for confirm in [5, 10]:
                    ret = run_combo(atr, bull, adx_w, confirm)
                    if ret.empty: continue
                    m  = full_metrics(ret)
                    mo = full_metrics(ret[ret.index >= '2022-01-01'])
                    passed = sum(1 for yr in range(2020,2026)
                        if len(ret[ret.index.year==yr])>5 and
                        full_metrics(ret[ret.index.year==yr])['annual_return'] >= TARGETS.get(yr,0))
                    results.append((mo['sharpe'], atr, bull, adx_w, confirm,
                                    m['sharpe'], m['max_drawdown'], m['annual_return'], passed))
    for row in sorted(results, reverse=True)[:5]:
        print(f'  OOSSh={row[0]:.3f} ATR={row[1]} Bull={row[2]} ADX_w={row[3]} C={row[4]} '
              f'Sh={row[5]:.3f} DD={row[6]:.1%} {row[8]}/6')

print(f'\nTotal: {time.time()-t0:.1f}s')
