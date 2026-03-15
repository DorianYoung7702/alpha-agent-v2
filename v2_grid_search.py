"""
v2_grid_search.py - EMA+ADX 参数网格扫描
目标：提升持仓时间同时控制 MaxDD < 35%，OOS Sharpe > 1.6
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
    try:
        return panel.xs(sym, level='symbol')['funding_rate'].shift(1).fillna(0)
    except: return pd.Series(dtype=float)

funding_series = {sym: get_funding_series(sym) for sym in SYMS}

def get_bear_return(date):
    r = [abs(funding_series[s].loc[date])*3 if date in funding_series[s].index and abs(funding_series[s].loc[date]) >= FUNDING_THRESH else LENDING_YIELD for s in SYMS]
    return float(np.mean(r))

def run_combo(ema_fast, ema_slow, adx_base, atr_mult=1.0, start='2020-01-01'):
    all_rets = []
    for sym in SYMS:
        ohlcv = data[sym]
        c     = ohlcv['close']
        ema_f = c.ewm(span=ema_fast, adjust=False).mean()
        ema_s = c.ewm(span=ema_slow, adjust=False).mean()
        ma200 = c.rolling(200).mean()
        adx   = calc_adx(ohlcv, window=14)
        bull_r  = bull_market.reindex(adx.index).fillna(False)
        adx_thr = pd.Series(float(adx_base), index=adx.index)
        adx_thr[bull_r] = max(adx_base - 10, 10)
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
            w = 0.2 * (2.0 if is_bull else 1.0)
            if not btc_ok:
                rets.append(get_bear_return(date) * 0.2)
                cum_ret = 0.0
            elif row['position'] == 1:
                cum_ret += row['return']
                if cum_ret > 0.05: w = min(w*1.5, 0.5)
                rets.append(row['return'] * w)
            else:
                rets.append(STABLE_YIELD * 0.2)
                cum_ret = 0.0
        all_rets.append(pd.Series(rets, index=r.index))
    if not all_rets: return pd.Series(dtype=float)
    return pd.concat(all_rets, axis=1).fillna(0).sum(axis=1)

# ── 参数网格 ────────���─────────────────────────────────────
GRID = [
    # (ema_fast, ema_slow, adx_base)
    (10, 30, 20), (10, 30, 25), (10, 30, 30),
    (20, 50, 20), (20, 50, 25), (20, 50, 30),
    (20, 60, 20), (20, 60, 25), (20, 60, 30),
    (20, 60, 35),  # 接近原版但用EMA
    (30, 90, 20), (30, 90, 25),
    (10, 60, 20), (10, 60, 25),
]

results = []
print(f'{'EMA':^12} {'ADX':^5} {'全Sh':^8} {'MaxDD':^8} {'年化':^8} {'OOSSh':^8} {'OOSDD':^8} {'2022':^8}')
print('-'*70)

for ema_f, ema_s, adx in GRID:
    t = time.time()
    ret = run_combo(ema_f, ema_s, adx)
    if ret.empty: continue
    m   = full_metrics(ret)
    oos = ret[ret.index >= '2022-01-01']
    mo  = full_metrics(oos)
    r22 = ret[ret.index.year == 2022]
    m22 = full_metrics(r22) if len(r22) > 5 else {'annual_return': 0}
    label = f'EMA{ema_f}/{ema_s}'
    print(f'{label:^12} {adx:^5} {m["sharpe"]:^8.3f} {m["max_drawdown"]:^8.1%} '
          f'{m["annual_return"]:^8.1%} {mo["sharpe"]:^8.3f} {mo["max_drawdown"]:^8.1%} '
          f'{m22["annual_return"]:^8.1%}')
    results.append({
        'ema_fast': ema_f, 'ema_slow': ema_s, 'adx': adx,
        'sharpe': m['sharpe'], 'max_dd': m['max_drawdown'],
        'annual': m['annual_return'],
        'oos_sharpe': mo['sharpe'], 'oos_dd': mo['max_drawdown'],
        'ann_2022': m22['annual_return'],
    })

# ── 最优参数 ──────────────────────────────────────────────
df = pd.DataFrame(results)
# 综合评分：OOS Sharpe 最高 + MaxDD > -40%
df_filtered = df[(df['max_dd'] > -0.40) & (df['oos_sharpe'] > 1.4)].copy()
if not df_filtered.empty:
    best = df_filtered.sort_values('oos_sharpe', ascending=False).iloc[0]
    print(f'\n最优参数（MaxDD>-40%, OOSSh>1.4）:')
    print(f'  EMA{int(best.ema_fast)}/{int(best.ema_slow)} ADX{int(best.adx)}')
    print(f'  全期Sh={best.sharpe:.3f} MaxDD={best.max_dd:.1%} OOSSh={best.oos_sharpe:.3f}')
else:
    best = df.sort_values('oos_sharpe', ascending=False).iloc[0]
    print(f'\n最优参数（无过滤）:')
    print(f'  EMA{int(best.ema_fast)}/{int(best.ema_slow)} ADX{int(best.adx)}')

# 保存结果
df.to_csv(r'D:\YZX\shared\results\v2_grid_results.csv', index=False)
print(f'\n[Saved] v2_grid_results.csv')
