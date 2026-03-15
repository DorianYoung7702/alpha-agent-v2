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

def run_combined(factors_list, top_n=3, adx_thresh=20, start=None, end=None):
    factor_vals = []
    for f in factors_list:
        fv = f.compute(panel)
        fv_z = fv.groupby(level='timestamp').transform(lambda x: (x-x.mean())/(x.std()+1e-10))
        factor_vals.append(fv_z)
    composite = pd.concat(factor_vals, axis=1).mean(axis=1)
    
    btc_result = engine.run(btc_signal, ohlcv_btc, start_date=start, end_date=end)
    if not isinstance(btc_result, pd.DataFrame) or btc_result.empty:
        return pd.DataFrame()
    
    combined_rets = []
    for date, row in btc_result.iterrows():
        if row['position'] == 1:
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

# 两个候选策略完整验证
CONFIGS = [
    ('hvol_10d_Top5',        [HistoricalVolatility(window=10)], 5),
    ('hvol_10d+dvol_Top3',   [HistoricalVolatility(window=10), DownsideVolatility(window=20)], 3),
    ('3factor_Top3',         [HistoricalVolatility(window=10), HistoricalVolatility(window=20), DownsideVolatility(window=20)], 3),
    ('hvol_10d_Top7',        [HistoricalVolatility(window=10)], 7),
]

for name, factors, top_n in CONFIGS:
    print(f'\n{"="*60}')
    print(f'策略: {name} (Top{top_n})')
    print(f'{"="*60}')
    r_full = run_combined(factors, top_n=top_n, start='2020-01-01')
    r_oos = run_combined(factors, top_n=top_n, start='2022-01-01')
    if isinstance(r_full, pd.DataFrame) and not r_full.empty:
        m = full_metrics(r_full['return'], factor_name=f'{name}_FULL')
        print_metrics(m)
        # 逐年
        print('逐年表现:')
        for yr in range(2020, 2026):
            ry = r_full[r_full.index.year == yr]
            if len(ry) > 10:
                my = full_metrics(ry['return'])
                print(f'  {yr}: Sharpe={my["sharpe"]:+.3f} MaxDD={my["max_drawdown"]:.1%} Annual={my["annual_return"]:+.1%}')
    if isinstance(r_oos, pd.DataFrame) and not r_oos.empty:
        m_oos = full_metrics(r_oos['return'], factor_name=f'{name}_OOS')
        print(f'\nOOS(2022-2026): Sharpe={m_oos["sharpe"]:.3f} MaxDD={m_oos["max_drawdown"]:.1%} Annual={m_oos["annual_return"]:+.1%}')
