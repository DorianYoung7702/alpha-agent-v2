import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from loguru import logger
logger.remove()
logger.add(sys.stdout, level='WARNING')

from data.fetcher import fetch_klines, fetch_all_symbols, fetch_funding_all, build_panel, SYMBOLS_TOP30
from factors.ts_signals_v2 import calc_adx
from factors.volatility import HistoricalVolatility
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics, print_metrics

STABLE_YIELD = 0.05 / 365

# 最���参数（来自 scan_multi.py）
OPTIMAL_PARAMS = {
    'BTCUSDT': dict(ma_f=200, adx_t=35, adx_w=20, atr_m=1.0),   # v1.0
    'ETHUSDT': dict(ma_f=200, adx_t=38, adx_w=14, atr_m=1.2),
    'SOLUSDT': dict(ma_f=200, adx_t=25, adx_w=14, atr_m=0.8),
    'BNBUSDT': dict(ma_f=150, adx_t=35, adx_w=14, atr_m=0.8),
    'ADAUSDT': dict(ma_f=150, adx_t=25, adx_w=14, atr_m=0.8),
    'AVAXUSDT': dict(ma_f=150, adx_t=35, adx_w=20, atr_m=0.8),
}

def make_signal(ohlcv, ma_f, adx_t, adx_w, fast=12, slow=26, sp=9):
    close = ohlcv['close']
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=sp, adjust=False).mean()
    ma = close.rolling(ma_f).mean()
    adx = calc_adx(ohlcv, window=adx_w)
    return ((hist > 0) & (close > ma) & (adx > adx_t)).astype(int).shift(1).fillna(0).astype(int)

# 加载 Top30 数据用于截面层
data_top30 = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMBOLS_TOP30, use_cache=True)
panel = build_panel(data_top30, funding_data=funding, min_history_days=60)
prices = panel['close'].unstack(level='symbol')
dates_list = sorted(prices.index)
hvol10 = HistoricalVolatility(window=10)
factor = hvol10.compute(panel)

# BTC ADX 用于截面层触发
ohlcv_btc = data_top30['BTCUSDT']
btc_adx = calc_adx(ohlcv_btc, window=20).shift(1).fillna(0)

def run_single(sym, weight=1.0, start=None, end=None, use_layer3=True):
    p = OPTIMAL_PARAMS[sym]
    if sym in data_top30:
        ohlcv = data_top30[sym]
    else:
        ohlcv = fetch_klines(sym, '1d', '2020-01-01', use_cache=True)
    
    if len(ohlcv) < 200:
        return pd.Series(dtype=float)
    
    sig = make_signal(ohlcv, p['ma_f'], p['adx_t'], p['adx_w'])
    engine = TimeSeriesEngine(atr_mult=p['atr_m'], use_atr_stop=True)
    r = engine.run(sig, ohlcv, start_date=start, end_date=end)
    if not isinstance(r, pd.DataFrame) or r.empty:
        return pd.Series(dtype=float)
    
    rets = []
    for date, row in r.iterrows():
        if row['position'] == 1:
            rets.append(row['return'] * weight)
        else:
            daily_ret = STABLE_YIELD
            # 只在BTC信号上用截面层（统一用BTC ADX判断市场状态）
            if use_layer3 and sym == 'BTCUSDT' and date in btc_adx.index and btc_adx.loc[date] < 20:
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

# 测试各组合
COMBOS = [
    ('BTC only (v1.0)',              ['BTCUSDT'], [1.0]),
    ('BTC+ETH',                     ['BTCUSDT','ETHUSDT'], [0.5,0.5]),
    ('BTC+ETH+BNB',                 ['BTCUSDT','ETHUSDT','BNBUSDT'], [1/3,1/3,1/3]),
    ('BTC+ETH+AVAX',                ['BTCUSDT','ETHUSDT','AVAXUSDT'], [1/3,1/3,1/3]),
    ('BTC+ETH+BNB+AVAX',            ['BTCUSDT','ETHUSDT','BNBUSDT','AVAXUSDT'], [0.25]*4),
    ('BTC+ETH+SOL+BNB+AVAX(5eq)',   ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','AVAXUSDT'], [0.2]*5),
    ('BTC+ETH+ADA+BNB+AVAX(5eq)',   ['BTCUSDT','ETHUSDT','ADAUSDT','BNBUSDT','AVAXUSDT'], [0.2]*5),
    ('ALL6_equal',                  ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','ADAUSDT','AVAXUSDT'], [1/6]*6),
]

print(f'{"组合":<35} {"全Sh":>7} {"MaxDD":>8} {"年化":>8} {"OOSSh":>8} {"OOSDD":>8}')
print('-'*78)
for name, syms, weights in COMBOS:
    all_rets = []
    for sym, w in zip(syms, weights):
        r = run_single(sym, weight=w, start='2020-01-01')
        if not r.empty:
            all_rets.append(r)
    if all_rets:
        combined = pd.concat(all_rets, axis=1).fillna(0).sum(axis=1)
        m = full_metrics(combined)
        oos = combined[combined.index >= '2022-01-01']
        m_oos = full_metrics(oos) if len(oos) > 50 else {}
        sh = m['sharpe']
        dd = m['max_drawdown']
        ann = m['annual_return']
        oos_sh = m_oos.get('sharpe',0)
        oos_dd = m_oos.get('max_drawdown',0)
        print(f'{name:<35} {sh:+>7.3f} {dd:+>8.1%} {ann:+>8.1%} {oos_sh:+>8.3f} {oos_dd:+>8.1%}')

# 最优组合逐年
print()
print('逐年（BTC+ETH+BNB+AVAX 4等权）:')
all_rets = [run_single(s, w, start='2020-01-01') 
            for s, w in zip(['BTCUSDT','ETHUSDT','BNBUSDT','AVAXUSDT'], [0.25]*4)]
combined = pd.concat(all_rets, axis=1).fillna(0).sum(axis=1)
for yr in range(2020, 2026):
    ry = combined[combined.index.year == yr]
    if len(ry) > 10:
        my = full_metrics(ry)
        print(f'  {yr}: Sharpe={my["sharpe"]:+.3f} MaxDD={my["max_drawdown"]:.1%} Annual={my["annual_return"]:+.1%}')
