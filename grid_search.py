import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from data.fetcher import fetch_all_symbols, SYMBOLS_TOP30
from factors.ts_signals_v2 import signal_macd_ma_filter
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics

data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
ohlcv = data['BTCUSDT']

results = []
# 精细扫参：MACD参数 × MA过滤 × ATR止损
for fast, slow, sig in [(12,26,9),(8,21,9),(10,30,9),(6,19,6),(15,35,10)]:
    for ma_f in [100, 150, 200]:
        for atr_m in [1.0, 1.5, 2.0]:
            try:
                signal = signal_macd_ma_filter(ohlcv, fast=fast, slow=slow, signal_period=sig, ma_filter=ma_f)
                engine = TimeSeriesEngine(atr_mult=atr_m, use_atr_stop=True)
                tr = engine.run(signal, ohlcv, start_date='2020-01-01', end_date='2023-12-31')
                if isinstance(tr, pd.DataFrame) and not tr.empty:
                    m = full_metrics(tr['return'], factor_name=f'macd_{fast}_{slow}_{sig}_ma{ma_f}_atr{atr_m}')
                    results.append((m['factor_name'], m['sharpe'], m['max_drawdown'], m['annual_return']))
            except: pass

results.sort(key=lambda x: x[1], reverse=True)
print('Top 15 (Train 2020-2023):')
for name, sh, dd, ann in results[:15]:
    print(f'  {name}: Sharpe={sh:.3f} MaxDD={dd:.1%} Annual={ann:.1%}')
