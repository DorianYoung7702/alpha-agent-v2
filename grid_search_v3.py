import sys, pandas as pd
sys.path.insert(0, 'src')
from data.fetcher import fetch_all_symbols, SYMBOLS_TOP30
from factors.ts_signals_v2 import calc_adx
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics

data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
ohlcv = data['BTCUSDT']

def signal_macd_ma_adx(ohlcv, fast=12, slow=26, signal_period=9, ma_filter=200, adx_threshold=25, adx_window=14):
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

results = []
# ���焦最佳参数 macd_12_26_9 + ma200 + adx30，细化其他维度
for adx_t in [25, 27, 30, 33, 35]:
    for adx_w in [10, 14, 20]:
        for atr_m in [0.8, 1.0, 1.2, 1.5]:
            try:
                signal = signal_macd_ma_adx(ohlcv, fast=12, slow=26, signal_period=9,
                    ma_filter=200, adx_threshold=adx_t, adx_window=adx_w)
                engine = TimeSeriesEngine(atr_mult=atr_m, use_atr_stop=True)
                tr = engine.run(signal, ohlcv, start_date='2020-01-01', end_date='2021-12-31')
                te = engine.run(signal, ohlcv, start_date='2022-01-01')
                if isinstance(tr, pd.DataFrame) and not tr.empty and isinstance(te, pd.DataFrame) and not te.empty:
                    tm = full_metrics(tr['return'])
                    tsm = full_metrics(te['return'])
                    name = f'adx{adx_t}w{adx_w}_atr{atr_m}'
                    results.append((
                        name,
                        tm.get('sharpe',0), tm.get('max_drawdown',-1),
                        tsm.get('sharpe',0), tsm.get('max_drawdown',-1),
                        tsm.get('annual_return',0),
                        (signal != 0).groupby(signal.index.year).mean().to_dict()
                    ))
            except: pass

results.sort(key=lambda x: x[3], reverse=True)
print('Top 15 by OOS Sharpe:')
print(f'{"Name":<25} {"TrSh":>6} {"TrDD":>7} {"TeSh":>6} {"TeDD":>7} {"TeAnn":>7}')
for r in results[:15]:
    print(f'{r[0]:<25} {r[1]:>6.3f} {r[2]:>7.1%} {r[3]:>6.3f} {r[4]:>7.1%} {r[5]:>7.1%}')
