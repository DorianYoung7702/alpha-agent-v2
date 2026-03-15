import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from loguru import logger
logger.remove()
logger.add(sys.stdout, level='WARNING')  # 只显示警告

from data.fetcher import fetch_klines, fetch_all_symbols, fetch_funding_all, build_panel, SYMBOLS_TOP30
from factors.ts_signals_v2 import calc_adx
from factors.volatility import HistoricalVolatility
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics

data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMBOLS_TOP30, use_cache=True)
panel = build_panel(data, funding_data=funding, min_history_days=60)
ohlcv_btc = data['BTCUSDT']

def make_signal(ohlcv):
    close = ohlcv['close']
    ema_f = close.ewm(span=12, adjust=False).mean()
    ema_s = close.ewm(span=26, adjust=False).mean()
    hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=9, adjust=False).mean()
    ma = close.rolling(200).mean()
    adx = calc_adx(ohlcv, window=20)
    return ((hist > 0) & (close > ma) & (adx > 35)).astype(int).shift(1).fillna(0).astype(int), adx.shift(1).fillna(0)

btc_signal, btc_adx = make_signal(ohlcv_btc)
engine = TimeSeriesEngine(atr_mult=1.0, use_atr_stop=True)
STABLE_YIELD = 0.05 / 365
prices = panel['close'].unstack(level='symbol')
dates_list = sorted(prices.index)
hvol10 = HistoricalVolatility(window=10)
factor = hvol10.compute(panel)

def run_v1(start=None, end=None, top_n=5, random_syms=False):
    r = engine.run(btc_signal, ohlcv_btc, start_date=start, end_date=end)
    if not isinstance(r, pd.DataFrame) or r.empty:
        return pd.Series(dtype=float)
    rets = []
    for date, row in r.iterrows():
        if row['position'] == 1:
            rets.append(row['return'])
        else:
            daily_ret = STABLE_YIELD
            adx_val = btc_adx.loc[date] if date in btc_adx.index else 99
            if adx_val < 20 and date in factor.index.get_level_values('timestamp'):
                try:
                    day_f = factor.xs(date, level='timestamp').dropna() if not random_syms else None
                    avail = [c for c in prices.columns if date in prices.index]
                    if avail and date in prices.index:
                        idx_d = dates_list.index(date) if date in dates_list else -1
                        if idx_d >= 0 and idx_d + 1 < len(dates_list):
                            nd = dates_list[idx_d + 1]
                            if random_syms:
                                syms = np.random.choice(avail, min(top_n, len(avail)), replace=False)
                            else:
                                syms = day_f.nlargest(top_n).index if day_f is not None and len(day_f) >= top_n*2 else []
                            if len(syms) > 0:
                                cr = sum((prices.loc[nd,s]-prices.loc[date,s])/prices.loc[date,s]
                                    for s in syms if s in prices.columns) / len(syms)
                                daily_ret = 0.5*cr + 0.5*STABLE_YIELD
                except: pass
            rets.append(daily_ret)
    return pd.Series(rets, index=r.index)

print('='*60)
print('v1.0 过拟合检验')
print('='*60)

# 1. 扩展���口
print('\n1. 扩展窗口验证:')
for ts, te in [('2022-01-01','2022-12-31'),('2023-01-01','2023-12-31'),('2024-01-01','2024-12-31'),('2025-01-01',None)]:
    r = run_v1(start=ts, end=te)
    if len(r) > 10:
        m = full_metrics(r)
        print(f'  {ts[:4]}: Sharpe={m["sharpe"]:+.3f} MaxDD={m["max_drawdown"]:.1%} Annual={m["annual_return"]:+.1%}')

# 2. 去掉2021
print('\n2. 去掉2021年（最强牛市）:')
r_full = run_v1(start='2020-01-01')
r_no21 = r_full[r_full.index.year != 2021]
m = full_metrics(r_no21)
print(f'  Sharpe={m["sharpe"]:+.3f} MaxDD={m["max_drawdown"]:.1%} Annual={m["annual_return"]:+.1%}')

# 3. 蒙特卡洛（20次，快速版）
print('\n3. 蒙特卡洛随机基准（20次）:')
np.random.seed(42)
random_sharpes = []
for _ in range(20):
    r = run_v1(start='2020-01-01', random_syms=True)
    if len(r) > 20:
        random_sharpes.append(full_metrics(r)['sharpe'])
if random_sharpes:
    print(f'  随机基准: mean={np.mean(random_sharpes):.3f} std={np.std(random_sharpes):.3f}')
    print(f'  v1.0 Sharpe=1.515 z-score: {(1.515-np.mean(random_sharpes))/(np.std(random_sharpes)+1e-10):.2f}σ')
    print(f'  v1.0 优于随机: {np.mean([s<1.515 for s in random_sharpes])*100:.0f}%')

print('\n完成')
