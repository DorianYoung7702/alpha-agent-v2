import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from data.fetcher import fetch_klines, fetch_all_symbols, fetch_funding_all, build_panel, SYMBOLS_TOP30
from factors.ts_signals_v2 import calc_adx
from factors.volatility import HistoricalVolatility
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics, print_metrics

# 加载数据
data_daily = {}
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    data_daily[sym] = fetch_klines(sym, '1d', '2020-01-01', use_cache=True)

data_top30 = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMBOLS_TOP30, use_cache=True)
panel = build_panel(data_top30, funding_data=funding, min_history_days=60)

STABLE_YIELD = 0.05 / 365
prices = panel['close'].unstack(level='symbol')
dates_list = sorted(prices.index)

def make_signal(ohlcv, fast=12, slow=26, sp=9, ma_f=200, adx_t=35, adx_w=20):
    close = ohlcv['close']
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=sp, adjust=False).mean()
    ma = close.rolling(ma_f).mean()
    adx = calc_adx(ohlcv, window=adx_w)
    result = ((hist > 0) & (close > ma) & (adx > adx_t)).astype(int)
    return result.shift(1).fillna(0).astype(int), adx.shift(1).fillna(0)

# 最优参数
PARAMS = {
    'BTCUSDT': dict(fast=12, slow=26, sp=9, ma_f=200, adx_t=35, adx_w=20),  # v1.0
    'ETHUSDT': dict(fast=12, slow=26, sp=9, ma_f=200, adx_t=35, adx_w=14),  # ETH最优
    'SOLUSDT': dict(fast=12, slow=26, sp=9, ma_f=100, adx_t=35, adx_w=20),  # SOL最优
}

hvol10 = HistoricalVolatility(window=10)
factor = hvol10.compute(panel)

def run_single_with_layer3(sym, weight=1.0, start=None, end=None):
    ohlcv = data_daily[sym]
    params = PARAMS[sym]
    sig, adx_s = make_signal(ohlcv, **params)
    engine = TimeSeriesEngine(atr_mult=1.0, use_atr_stop=True)
    r = engine.run(sig, ohlcv, start_date=start, end_date=end)
    if not isinstance(r, pd.DataFrame) or r.empty:
        return pd.Series(dtype=float)
    
    rets = []
    for date, row in r.iterrows():
        if row['position'] == 1:
            rets.append(row['return'] * weight)
        else:
            daily_ret = STABLE_YIELD
            adx_val = adx_s.loc[date] if date in adx_s.index else 99
            if adx_val < 20 and date in factor.index.get_level_values('timestamp'):
                try:
                    day_f = factor.xs(date, level='timestamp').dropna()
                    if len(day_f) >= 10:
                        top = day_f.nlargest(5)
                        if date in prices.index:
                            idx = dates_list.index(date)
                            if idx + 1 < len(dates_list):
                                nd = dates_list[idx + 1]
                                cr = sum((prices.loc[nd,s]-prices.loc[date,s])/prices.loc[date,s]
                                    for s in top.index if s in prices.columns) / len(top)
                                daily_ret = 0.5*cr + 0.5*STABLE_YIELD
                except: pass
            rets.append(daily_ret * weight)
    return pd.Series(rets, index=r.index)

# 测试不同组合
configs = [
    ('BTC only (v1.0)',     ['BTCUSDT'], [1.0]),
    ('BTC+ETH (50/50)',     ['BTCUSDT','ETHUSDT'], [0.5,0.5]),
    ('BTC+ETH+SOL (equal)', ['BTCUSDT','ETHUSDT','SOLUSDT'], [1/3,1/3,1/3]),
    ('BTC+SOL (50/50)',     ['BTCUSDT','SOLUSDT'], [0.5,0.5]),
]

print(f'{"策略":<30} {"全Sh":>7} {"MaxDD":>8} {"年���":>8} {"OOSSh":>8} {"OOSDD":>8}')
print('-'*73)

for name, syms, weights in configs:
    all_rets = []
    for sym, w in zip(syms, weights):
        r = run_single_with_layer3(sym, weight=w, start='2020-01-01')
        if not r.empty:
            all_rets.append(r)
    if all_rets:
        combined = pd.concat(all_rets, axis=1).fillna(0).sum(axis=1)
        m = full_metrics(combined)
        oos = combined[combined.index >= '2022-01-01']
        m_oos = full_metrics(oos) if len(oos) > 20 else {}
        print(f'{name:<30} {m["sharpe"]:+>7.3f} {m["max_drawdown"]:+>8.1%} {m["annual_return"]:+>8.1%} '
              f'{m_oos.get("sharpe",0):+>8.3f} {m_oos.get("max_drawdown",0):+>8.1%}')

# 最优组合逐年分析
print()
print('BTC+ETH+SOL 等权逐年:')
all_rets = [run_single_with_layer3(s, w, start='2020-01-01') 
            for s, w in zip(['BTCUSDT','ETHUSDT','SOLUSDT'], [1/3,1/3,1/3])]
combined = pd.concat(all_rets, axis=1).fillna(0).sum(axis=1)
for yr in range(2020, 2026):
    ry = combined[combined.index.year == yr]
    if len(ry) > 10:
        my = full_metrics(ry)
        print(f'  {yr}: Sharpe={my["sharpe"]:+.3f} MaxDD={my["max_drawdown"]:.1%} Annual={my["annual_return"]:+.1%}')
