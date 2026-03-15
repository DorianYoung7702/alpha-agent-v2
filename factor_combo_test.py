import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from data.fetcher import fetch_all_symbols, fetch_funding_all, build_panel, SYMBOLS_TOP30
from factors.volatility import HistoricalVolatility, DownsideVolatility
from factors.ts_signals_v2 import calc_adx
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics, print_metrics

data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMBOLS_TOP30, use_cache=True)
panel = build_panel(data, funding_data=funding, min_history_days=60)
ohlcv_btc = data['BTCUSDT']

def signal_macd_ma_adx(ohlcv):
    from factors.ts_signals_v2 import calc_adx
    close = ohlcv['close']
    ema_f = close.ewm(span=12, adjust=False).mean()
    ema_s = close.ewm(span=26, adjust=False).mean()
    hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=9, adjust=False).mean()
    ma = close.rolling(200).mean()
    adx = calc_adx(ohlcv, window=20)
    result = ((hist > 0) & (close > ma) & (adx > 35)).astype(int)
    return result.shift(1).fillna(0).astype(int), adx.shift(1).fillna(0)

btc_signal, btc_adx = signal_macd_ma_adx(ohlcv_btc)
engine = TimeSeriesEngine(atr_mult=1.0, use_atr_stop=True)
STABLE_YIELD = 0.05 / 365
prices = panel['close'].unstack(level='symbol')
dates_list = sorted(prices.index)

# 测试不同截���因子组合
FACTOR_CONFIGS = [
    ('hvol_20d', [HistoricalVolatility(window=20)]),
    ('hvol_10d', [HistoricalVolatility(window=10)]),
    ('hvol_10d+20d', [HistoricalVolatility(window=10), HistoricalVolatility(window=20)]),
    ('hvol_10d+downvol', [HistoricalVolatility(window=10), DownsideVolatility(window=20)]),
    ('3factor_combo', [HistoricalVolatility(window=10), HistoricalVolatility(window=20), DownsideVolatility(window=20)]),
    ('top5_hvol10', [HistoricalVolatility(window=10)]),  # top5 instead of top3
]

def run_combined_with_factor(factors_list, top_n=3, adx_thresh=20):
    # 计算合成因子
    factor_vals = []
    for f in factors_list:
        try:
            fv = f.compute(panel)
            fv_z = fv.groupby(level='timestamp').transform(lambda x: (x-x.mean())/(x.std()+1e-10))
            factor_vals.append(fv_z)
        except: pass
    if not factor_vals:
        return pd.DataFrame()
    composite = pd.concat(factor_vals, axis=1).mean(axis=1)
    
    btc_result = engine.run(btc_signal, ohlcv_btc, start_date='2020-01-01')
    if not isinstance(btc_result, pd.DataFrame) or btc_result.empty:
        return pd.DataFrame()
    
    combined_rets = []
    for date, row in btc_result.iterrows():
        btc_pos = row['position']
        if btc_pos == 1:
            combined_rets.append({'timestamp': date, 'return': row['return'], 'mode': 'btc'})
        else:
            daily_ret = STABLE_YIELD
            mode = 'stable'
            adx_val = btc_adx.loc[date] if date in btc_adx.index else 99
            if adx_val < adx_thresh and date in composite.index.get_level_values('timestamp'):
                try:
                    day_f = composite.xs(date, level='timestamp').dropna()
                    if len(day_f) >= top_n * 2:
                        top = day_f.nlargest(top_n)
                        if date in prices.index:
                            idx = dates_list.index(date)
                            if idx + 1 < len(dates_list):
                                nd = dates_list[idx + 1]
                                cr = sum((prices.loc[nd,s]-prices.loc[date,s])/prices.loc[date,s]
                                    for s in top.index if s in prices.columns) / len(top)
                                daily_ret = 0.5*cr + 0.5*STABLE_YIELD
                                mode = 'cross'
                except: pass
            combined_rets.append({'timestamp': date, 'return': daily_ret, 'mode': mode})
    return pd.DataFrame(combined_rets).set_index('timestamp')

print(f'{'因子配置':<25} {'全期Sharpe':>11} {'MaxDD':>8} {'年化':>8} {'OOSSharpe':>11} {'OOSMaxDD':>9}')
print('-'*75)

for name, factors_list in FACTOR_CONFIGS[:5]:
    top_n = 5 if name == 'top5_hvol10' else 3
    r = run_combined_with_factor(factors_list, top_n=top_n)
    if isinstance(r, pd.DataFrame) and not r.empty:
        m = full_metrics(r['return'])
        oos = r[r.index >= '2022-01-01']
        m_oos = full_metrics(oos['return']) if len(oos) > 20 else {}
        print(f'{name:<25} {m["sharpe"]:+>11.3f} {m["max_drawdown"]:+>8.1%} {m["annual_return"]:+>8.1%} '
              f'{m_oos.get("sharpe",0):+>11.3f} {m_oos.get("max_drawdown",0):+>9.1%}')

# Top5 单独测试
print()
print('Top5 vs Top3 对比（hvol_10d）:')
for tn in [3, 5, 7]:
    r = run_combined_with_factor([HistoricalVolatility(window=10)], top_n=tn)
    if isinstance(r, pd.DataFrame) and not r.empty:
        m = full_metrics(r['return'])
        oos = r[r.index >= '2022-01-01']
        m_oos = full_metrics(oos['return'])
        print(f'  Top{tn}: Sharpe={m["sharpe"]:.3f} MaxDD={m["max_drawdown"]:.1%} OOS={m_oos["sharpe"]:.3f}')
