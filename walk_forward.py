import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from data.fetcher import fetch_all_symbols, SYMBOLS_TOP30
from factors.ts_signals_v2 import calc_adx
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics, print_metrics

data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
ohlcv = data['BTCUSDT']

def signal_macd_ma_adx(ohlcv, fast=12, slow=26, signal_period=9, ma_filter=200, adx_threshold=35, adx_window=20):
    close = ohlcv['close']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    hist = macd_line - sig_line
    ma = close.rolling(ma_filter).mean()
    adx = calc_adx(ohlcv, window=adx_window)
    macd_sig = (hist > 0).astype(int)
    trend_filter = (close > ma).astype(int).shift(1).fillna(0)
    adx_filter = (adx > adx_threshold).astype(int).shift(1).fillna(0)
    result = macd_sig * trend_filter * adx_filter
    return result.shift(1).fillna(0).astype(int)

signal = signal_macd_ma_adx(ohlcv)
engine = TimeSeriesEngine(atr_mult=1.0, use_atr_stop=True)

print('='*60)
print('最佳策略: MACD(12,26,9) + MA200 + ADX35(w20) + ATR1.0止损')
print('='*60)

# Walk-forward 验证（每年一个窗口）
periods = [
    ('2020', '2020-01-01', '2020-12-31'),
    ('2021', '2021-01-01', '2021-12-31'),
    ('2022', '2022-01-01', '2022-12-31'),
    ('2023', '2023-01-01', '2023-12-31'),
    ('2024', '2024-01-01', '2024-12-31'),
    ('2025', '2025-01-01', None),
]
print('\nWalk-forward 逐年表现:')
for name, start, end in periods:
    r = engine.run(signal, ohlcv, start_date=start, end_date=end)
    if isinstance(r, pd.DataFrame) and not r.empty:
        m = full_metrics(r['return'])
        hold = (r['position']==1).mean()
        print(f'  {name}: Sharpe={m["sharpe"]:+.3f} MaxDD={m["max_drawdown"]:.1%} Annual={m["annual_return"]:+.1%} Hold={hold:.0%}')

print()
# 全样本
full = engine.run(signal, ohlcv, start_date='2020-01-01')
m = full_metrics(full['return'])
print_metrics(m)
print(f'全期持仓比例: {(full["position"]==1).mean():.1%}')
