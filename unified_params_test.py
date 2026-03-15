import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from loguru import logger
logger.remove()
logger.add(sys.stdout, level='WARNING')

from data.fetcher import fetch_all_symbols, fetch_funding_all, build_panel, SYMBOLS_TOP30
from factors.ts_signals_v2 import calc_adx
from factors.volatility import HistoricalVolatility
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics

STABLE_YIELD = 0.05 / 365
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'AVAXUSDT']  # ADA排除（数据异常）

# 统一参数（BTC v1.0参数，不做品种定制）
UNIFIED_PARAMS = dict(ma_f=200, adx_t=35, adx_w=20, atr_m=1.0)

def make_signal(ohlcv, ma_f=200, adx_t=35, adx_w=20):
    close = ohlcv['close']
    ema_f = close.ewm(span=12, adjust=False).mean()
    ema_s = close.ewm(span=26, adjust=False).mean()
    hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=9, adjust=False).mean()
    ma = close.rolling(ma_f).mean()
    adx = calc_adx(ohlcv, window=adx_w)
    return ((hist > 0) & (close > ma) & (adx > adx_t)).astype(int).shift(1).fillna(0).astype(int)

data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMBOLS_TOP30, use_cache=True)
panel = build_panel(data, funding_data=funding, min_history_days=60)
prices = panel['close'].unstack(level='symbol')
dates_list = sorted(prices.index)
hvol10 = HistoricalVolatility(window=10)
factor = hvol10.compute(panel)
btc_adx = calc_adx(data['BTCUSDT'], window=20).shift(1).fillna(0)

def run_sym(sym, weight, start=None):
    if sym not in data:
        return pd.Series(dtype=float)
    ohlcv = data[sym]
    p = UNIFIED_PARAMS
    sig = make_signal(ohlcv, p['ma_f'], p['adx_t'], p['adx_w'])
    engine = TimeSeriesEngine(atr_mult=p['atr_m'], use_atr_stop=True)
    r = engine.run(sig, ohlcv, start_date=start)
    if not isinstance(r, pd.DataFrame) or r.empty:
        return pd.Series(dtype=float)
    rets = []
    for date, row in r.iterrows():
        if row['position'] == 1:
            rets.append(row['return'] * weight)
        else:
            daily_ret = STABLE_YIELD
            if sym == 'BTCUSDT' and date in btc_adx.index and btc_adx.loc[date] < 20:
                try:
                    day_f = factor.xs(date, level='timestamp').dropna()
                    if len(day_f) >= 10 and date in prices.index:
                        idx = dates_list.index(date)
                        if idx + 1 < len(dates_list):
                            nd = dates_list[idx + 1]
                            top = day_f.nlargest(5)
                            cr = sum((prices.loc[nd,s]-prices.loc[date,s])/prices.loc[date,s]
                                for s in top.index if s in prices.columns) / len(top)
                            daily_ret = 0.5*cr + 0.5*STABLE_YIELD
                except: pass
            rets.append(daily_ret * weight)
    return pd.Series(rets, index=r.index)

print('���一参数测试（所有品种用BTC v1.0参数 adx35w20_ma200_atr1.0）')
print('='*65)

# 逐步添加品种
for n in range(1, len(SYMBOLS)+1):
    syms = SYMBOLS[:n]
    w = 1.0 / n
    all_rets = [run_sym(s, w, start='2020-01-01') for s in syms]
    combined = pd.concat([r for r in all_rets if not r.empty], axis=1).fillna(0).sum(axis=1)
    oos = combined[combined.index >= '2022-01-01']
    m = full_metrics(combined)
    m_oos = full_metrics(oos) if len(oos) > 50 else {}
    print(f'{"_".join(syms):<45} Sh={m["sharpe"]:+.3f} DD={m["max_drawdown"]:.1%} OOS={m_oos.get("sharpe",0):+.3f} OOSDD={m_oos.get("max_drawdown",0):.1%}')

# 详细逐年（5币等权）
print()
print('5币等权（统一参数）逐年：')
all_rets = [run_sym(s, 0.2, start='2020-01-01') for s in SYMBOLS]
combined = pd.concat([r for r in all_rets if not r.empty], axis=1).fillna(0).sum(axis=1)
for yr in range(2020, 2026):
    ry = combined[combined.index.year == yr]
    if len(ry) > 10:
        my = full_metrics(ry)
        print(f'  {yr}: Sh={my["sharpe"]:+.3f} DD={my["max_drawdown"]:.1%} Ann={my["annual_return"]:+.1%}')
m5 = full_metrics(combined)
oos5 = combined[combined.index >= '2022-01-01']
m5oos = full_metrics(oos5)
print(f'  全期: Sh={m5["sharpe"]:+.3f} DD={m5["max_drawdown"]:.1%} Ann={m5["annual_return"]:+.1%}')
print(f'  OOS:  Sh={m5oos["sharpe"]:+.3f} DD={m5oos["max_drawdown"]:.1%} Ann={m5oos["annual_return"]:+.1%}')
