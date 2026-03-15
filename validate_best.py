import sys, pandas as pd
sys.path.insert(0, 'src')
from data.fetcher import fetch_all_symbols, SYMBOLS_TOP30
from factors.ts_signals_v2 import signal_macd_ma_filter
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics, print_metrics

data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
ohlcv = data['BTCUSDT']

# 最佳参数
signal = signal_macd_ma_filter(ohlcv, fast=12, slow=26, signal_period=9, ma_filter=200)
engine = TimeSeriesEngine(atr_mult=1.0, use_atr_stop=True)

# 样本内
train = engine.run(signal, ohlcv, start_date='2020-01-01', end_date='2023-12-31')
train_m = full_metrics(train['return'], factor_name='BTC_macd_ma200_atr1.0_TRAIN')
print_metrics(train_m)

# 样本外
test = engine.run(signal, ohlcv, start_date='2024-01-01')
test_m = full_metrics(test['return'], factor_name='BTC_macd_ma200_atr1.0_TEST')
print_metrics(test_m)

# 全周期
full = engine.run(signal, ohlcv, start_date='2020-01-01')
full_m = full_metrics(full['return'], factor_name='BTC_macd_ma200_atr1.0_FULL')
print_metrics(full_m)

# 仓位统计
print(f'样本内持仓���数: {(train["position"]==1).sum()}/{len(train)} ({(train["position"]==1).mean():.1%})')
print(f'样本外持仓天数: {(test["position"]==1).sum()}/{len(test)} ({(test["position"]==1).mean():.1%})')
