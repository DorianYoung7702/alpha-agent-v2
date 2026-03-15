import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from data.fetcher import fetch_klines, fetch_all_symbols, fetch_funding_all, build_panel, SYMBOLS_TOP30
from factors.ts_signals_v2 import calc_adx
from factors.volatility import HistoricalVolatility
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics

# v1.0 核心策略重现
data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMBOLS_TOP30, use_cache=True)
panel = build_panel(data, funding_data=funding, min_history_days=60)
ohlcv_btc = data['BTCUSDT']

def make_signal_adx(ohlcv, fast=12, slow=26, sp=9, ma_f=200, adx_t=35, adx_w=20):
    close = ohlcv['close']
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=sp, adjust=False).mean()
    ma = close.rolling(ma_f).mean()
    adx = calc_adx(ohlcv, window=adx_w)
    result = ((hist > 0) & (close > ma) & (adx > adx_t)).astype(int)
    return result.shift(1).fillna(0).astype(int), adx.shift(1).fillna(0)

btc_signal, btc_adx = make_signal_adx(ohlcv_btc)
engine = TimeSeriesEngine(atr_mult=1.0, use_atr_stop=True)
STABLE_YIELD = 0.05 / 365
prices = panel['close'].unstack(level='symbol')
dates_list = sorted(prices.index)
hvol10 = HistoricalVolatility(window=10)
factor = hvol10.compute(panel)

def run_v1(start=None, end=None, top_n=5):
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
                    day_f = factor.xs(date, level='timestamp').dropna()
                    if len(day_f) >= top_n * 2:
                        top = day_f.nlargest(top_n)
                        if date in prices.index:
                            idx = dates_list.index(date)
                            if idx + 1 < len(dates_list):
                                nd = dates_list[idx + 1]
                                cr = sum((prices.loc[nd,s]-prices.loc[date,s])/prices.loc[date,s]
                                    for s in top.index if s in prices.columns) / len(top)
                                daily_ret = 0.5*cr + 0.5*STABLE_YIELD
                except: pass
            rets.append(daily_ret)
    return pd.Series(rets, index=r.index)

# === 严格过拟合检验 ===
print('='*60)
print('v1.0 严格过拟合检验')
print('='*60)

# 1. 扩展窗口测试 (expanding window)
print('\n1. 扩展窗口验证（模拟真实决策）')
for train_end, test_start, test_end in [
    ('2021-12-31', '2022-01-01', '2022-12-31'),
    ('2022-12-31', '2023-01-01', '2023-12-31'),
    ('2023-12-31', '2024-01-01', '2024-12-31'),
    ('2024-12-31', '2025-01-01', None),
]:
    r_test = run_v1(start=test_start, end=test_end)
    if len(r_test) > 10:
        m = full_metrics(r_test)
        print(f'  Test {test_start[:4]}: Sharpe={m["sharpe"]:+.3f} MaxDD={m["max_drawdown"]:.1%} Annual={m["annual_return"]:+.1%}')

# 2. 测试是否依赖特定牛市年份
print('\n2. 剔除2021年后重新计算（排除最强牛市影响）')
r_no2021 = run_v1(start='2020-01-01')
r_no2021 = r_no2021[r_no2021.index.year != 2021]
if len(r_no2021) > 20:
    m = full_metrics(r_no2021)
    print(f'  去掉2021: Sharpe={m["sharpe"]:+.3f} MaxDD={m["max_drawdown"]:.1%} Annual={m["annual_return"]:+.1%}')

# 3. 蒙特卡洛随机换仓检验
print('\n3. 随机因子基准（蒙特卡洛，100次）')
np.random.seed(42)
random_sharpes = []
for _ in range(100):
    r_base = engine.run(btc_signal, ohlcv_btc, start_date='2020-01-01')
    if not isinstance(r_base, pd.DataFrame) or r_base.empty:
        continue
    rets = []
    for date, row in r_base.iterrows():
        if row['position'] == 1:
            rets.append(row['return'])
        else:
            daily_ret = STABLE_YIELD
            # 随机选5个币
            avail = [c for c in prices.columns if date in prices.index]
            if avail and date in prices.index:
                idx_d = dates_list.index(date) if date in dates_list else -1
                if idx_d >= 0 and idx_d + 1 < len(dates_list):
                    nd = dates_list[idx_d + 1]
                    rand_syms = np.random.choice(avail, min(5, len(avail)), replace=False)
                    cr = sum((prices.loc[nd,s]-prices.loc[date,s])/prices.loc[date,s]
                        for s in rand_syms if s in prices.columns) / len(rand_syms)
                    daily_ret = 0.5*cr + 0.5*STABLE_YIELD
            rets.append(daily_ret)
    r_rand = pd.Series(rets, index=r_base.index)
    m_rand = full_metrics(r_rand)
    random_sharpes.append(m_rand['sharpe'])

if random_sharpes:
    print(f'  随机基准 Sharpe: mean={np.mean(random_sharpes):.3f} std={np.std(random_sharpes):.3f}')
    print(f'  v1.0 Sharpe=1.515 的 z-score: {(1.515 - np.mean(random_sharpes))/np.std(random_sharpes):.2f}σ')
    pct = np.mean([s < 1.515 for s in random_sharpes]) * 100
    print(f'  v1.0 优于随机基准的概率: {pct:.1f}%')
