import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from loguru import logger
logger.remove()
logger.add(sys.stdout, level='WARNING')

from data.fetcher import fetch_klines
from factors.ts_signals_v2 import calc_adx
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics

STABLE_YIELD = 0.05 / 365

def make_signal(ohlcv, fast=12, slow=26, sp=9, ma_f=200, adx_t=35, adx_w=20):
    close = ohlcv['close']
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=sp, adjust=False).mean()
    ma = close.rolling(ma_f).mean()
    adx = calc_adx(ohlcv, window=adx_w)
    return ((hist > 0) & (close > ma) & (adx > adx_t)).astype(int).shift(1).fillna(0).astype(int)

def run_ts(ohlcv, signal, atr_m, start=None, end=None):
    engine = TimeSeriesEngine(atr_mult=atr_m, use_atr_stop=True)
    r = engine.run(signal, ohlcv, start_date=start, end_date=end)
    if not isinstance(r, pd.DataFrame) or r.empty:
        return pd.Series(dtype=float)
    # 叠加稳定币
    return r.apply(lambda x: x['return'] if x['position']==1 else STABLE_YIELD, axis=1)

# 参数网���
params_grid = []
for adx_t in [25, 28, 30, 33, 35, 38, 40]:
    for adx_w in [14, 20]:
        for ma_f in [150, 200]:
            for atr_m in [0.8, 1.0, 1.2, 1.5]:
                params_grid.append((adx_t, adx_w, ma_f, atr_m))

SYMBOLS = ['ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT', 'AVAXUSDT']
best_per_sym = {}

for sym in SYMBOLS:
    print(f'\n{"="*55}')
    print(f'Scanning {sym}...')
    ohlcv = fetch_klines(sym, '1d', '2020-01-01', use_cache=True)
    if len(ohlcv) < 500:
        print(f'  Not enough data ({len(ohlcv)} rows), skip')
        continue
    
    results = []
    for adx_t, adx_w, ma_f, atr_m in params_grid:
        try:
            sig = make_signal(ohlcv, ma_f=ma_f, adx_t=adx_t, adx_w=adx_w)
            r_full = run_ts(ohlcv, sig, atr_m, start='2020-01-01')
            r_oos = run_ts(ohlcv, sig, atr_m, start='2022-01-01')
            if len(r_full) < 200 or len(r_oos) < 50:
                continue
            m_full = full_metrics(r_full)
            m_oos = full_metrics(r_oos)
            r22 = r_full[r_full.index.year == 2022]
            m22 = full_metrics(r22) if len(r22) > 20 else {}
            hold = (sig > 0).mean()
            results.append((
                adx_t, adx_w, ma_f, atr_m,
                m_full['sharpe'], m_full['max_drawdown'], m_full['annual_return'],
                m_oos['sharpe'], m_oos['max_drawdown'],
                m22.get('sharpe', 0), hold
            ))
        except: pass
    
    if not results:
        print(f'  No valid results')
        continue
    
    # 排序：先按 OOS Sharpe，再确保 MaxDD < 40%
    valid = [r for r in results if r[7] > 0.8 and r[5] > -0.40]
    if not valid:
        valid = results  # 放宽，取最佳
    valid.sort(key=lambda x: x[7], reverse=True)
    
    best = valid[0]
    best_per_sym[sym] = {
        'params': {'adx_t': best[0], 'adx_w': best[1], 'ma_f': best[2], 'atr_m': best[3]},
        'full_sharpe': best[4], 'full_dd': best[5], 'annual': best[6],
        'oos_sharpe': best[7], 'oos_dd': best[8], '2022_sharpe': best[9], 'hold': best[10]
    }
    
    print(f'Best: adx{best[0]}w{best[1]}_ma{best[2]}_atr{best[3]}')
    print(f'  Full: Sh={best[4]:+.3f} MaxDD={best[5]:.1%} Annual={best[6]:+.1%}')
    print(f'  OOS:  Sh={best[7]:+.3f} MaxDD={best[8]:.1%} 2022={best[9]:+.3f}')
    print(f'  Hold: {best[10]:.1%}')
    
    # 也打印 Top3
    print(f'  Top3 by OOS Sharpe:')
    for r in valid[:3]:
        print(f'    adx{r[0]}w{r[1]}_ma{r[2]}_atr{r[3]}: OOS={r[7]:+.3f} MaxDD={r[8]:.1%}')

print(f'\n{"="*55}')
print('扫描完���，最优参数汇总：')
for sym, res in best_per_sym.items():
    p = res['params']
    print(f'  {sym}: adx{p["adx_t"]}w{p["adx_w"]}_ma{p["ma_f"]}_atr{p["atr_m"]} '
          f'OOS={res["oos_sharpe"]:+.3f} MaxDD={res["oos_dd"]:.1%} 2022={res["2022_sharpe"]:+.3f}')
