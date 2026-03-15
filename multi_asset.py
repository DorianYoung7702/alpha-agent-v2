import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from data.fetcher import fetch_all_symbols, SYMBOLS_TOP30
from factors.ts_signals_v2 import calc_adx
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics, print_metrics

data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)

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

# 三币各自���号
symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
weight = 1.0 / len(symbols)  # 各33%
engine = TimeSeriesEngine(atr_mult=1.0, use_atr_stop=True)

all_returns = {}
for sym in symbols:
    if sym not in data:
        print(f'No data for {sym}')
        continue
    ohlcv = data[sym]
    signal = signal_macd_ma_adx(ohlcv)
    result = engine.run(signal, ohlcv, start_date='2020-01-01')
    if isinstance(result, pd.DataFrame) and not result.empty:
        all_returns[sym] = result['return'] * weight
        print(f'{sym}: {len(result)} days, hold={result["position"].mean():.1%}')

# 合并组合收益
if all_returns:
    combined = pd.concat(all_returns.values(), axis=1).fillna(0).sum(axis=1)
    combined.name = 'portfolio_return'
    
    m = full_metrics(combined, factor_name='BTC_ETH_SOL_Portfolio_FULL')
    print_metrics(m)
    
    # 逐年
    print('\nWalk-forward 逐年:')
    for year in range(2020, 2026):
        yr = combined[combined.index.year == year]
        if len(yr) > 10:
            ym = full_metrics(yr)
            print(f'  {year}: Sharpe={ym["sharpe"]:+.3f} MaxDD={ym["max_drawdown"]:.1%} Annual={ym["annual_return"]:+.1%}')
    
    # 样本内外
    train = combined[combined.index < '2022-01-01']
    test = combined[combined.index >= '2022-01-01']
    tm = full_metrics(train, factor_name='Portfolio_TRAIN_2020-2021')
    tsm = full_metrics(test, factor_name='Portfolio_TEST_2022-2026')
    print()
    print_metrics(tm)
    print_metrics(tsm)
    
    # 持仓统计
    print('\n各币种持仓比例:')
    for sym, ret in all_returns.items():
        full_r = engine.run(signal_macd_ma_adx(data[sym]), data[sym], start_date='2020-01-01')
        if isinstance(full_r, pd.DataFrame):
            print(f'  {sym}: {full_r["position"].mean():.1%}')
