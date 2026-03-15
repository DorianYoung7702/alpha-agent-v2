import sys, pandas as pd
sys.path.insert(0, 'src')
from data.fetcher import fetch_all_symbols, SYMBOLS_TOP30
from factors.ts_signals_v2 import signal_macd_ma_filter, calc_adx
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics, print_metrics

data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
ohlcv = data['BTCUSDT']

def signal_macd_ma_adx(ohlcv, fast=12, slow=26, signal_period=9, ma_filter=200, adx_threshold=25):
    """MACD + MA���滤 + ADX震荡市过滤"""
    import numpy as np
    close = ohlcv['close']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    hist = macd_line - sig_line
    ma = close.rolling(ma_filter).mean()
    adx = calc_adx(ohlcv, window=14)
    
    macd_sig = (hist > 0).astype(int)
    trend_filter = (close > ma).astype(int).shift(1).fillna(0)
    adx_filter = (adx > adx_threshold).astype(int).shift(1).fillna(0)
    
    result = macd_sig * trend_filter * adx_filter
    return result.shift(1).fillna(0).astype(int)

results = []
for fast, slow, sig in [(12,26,9),(10,30,9),(8,21,9)]:
    for adx_t in [20, 25, 30]:
        for atr_m in [1.0, 1.5, 2.0]:
            for ma_f in [100, 150, 200]:
                try:
                    signal = signal_macd_ma_adx(ohlcv, fast=fast, slow=slow,
                        signal_period=sig, ma_filter=ma_f, adx_threshold=adx_t)
                    engine = TimeSeriesEngine(atr_mult=atr_m, use_atr_stop=True)
                    
                    # 训练：2020-2021（纯牛市）
                    tr = engine.run(signal, ohlcv, start_date='2020-01-01', end_date='2021-12-31')
                    # 测试：2022-2024（熊市+震荡）
                    te = engine.run(signal, ohlcv, start_date='2022-01-01')
                    
                    if isinstance(tr, pd.DataFrame) and not tr.empty and isinstance(te, pd.DataFrame) and not te.empty:
                        tm = full_metrics(tr['return'])
                        tsm = full_metrics(te['return'])
                        name = f'macd_{fast}_{slow}_ma{ma_f}_adx{adx_t}_atr{atr_m}'
                        results.append((
                            name,
                            tm.get('sharpe',0), tm.get('max_drawdown',-1),
                            tsm.get('sharpe',0), tsm.get('max_drawdown',-1),
                            tsm.get('annual_return',0)
                        ))
                except Exception as e:
                    pass

# 按样本外Sharpe排序
results.sort(key=lambda x: x[3], reverse=True)
print('Top 20 by OOS Sharpe (Test: 2022-2024):')
print(f'{"Name":<45} {"TrainSh":>8} {"TrainDD":>8} {"TestSh":>8} {"TestDD":>8} {"TestAnn":>8}')
for r in results[:20]:
    print(f'{r[0]:<45} {r[1]:>8.3f} {r[2]:>8.1%} {r[3]:>8.3f} {r[4]:>8.1%} {r[5]:>8.1%}')
