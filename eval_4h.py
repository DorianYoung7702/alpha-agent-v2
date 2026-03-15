import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from data.fetcher import fetch_klines
from factors.ts_signals_v2 import calc_adx
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics, print_metrics

# 拉取全部4小时数据
print('Fetching 4h data...')
symbols = ['BTCUSDT', 'ETHUSDT']
data = {}
for sym in symbols:
    df = fetch_klines(sym, '4h', '2020-01-01', use_cache=True)
    data[sym] = df
    print(f'{sym}: {len(df)} rows')

ohlcv = data['BTCUSDT']
print(f'BTC 4h: {len(ohlcv)} bars ({len(ohlcv)/6:.0f} trading days equiv)')

# 适配4小时线的信号参数
# 日线 MACD(12,26,9) 对应 4h 约乘以 6
# 但先测原参数，再测适配参数
def signal_4h(ohlcv, fast=12, slow=26, sp=9, ma_f=200, adx_t=35, adx_w=20):
    close = ohlcv['close']
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=sp, adjust=False).mean()
    ma = close.rolling(ma_f).mean()
    adx = calc_adx(ohlcv, window=adx_w)
    result = ((hist > 0) & (close > ma) & (adx > adx_t)).astype(int)
    return result.shift(1).fillna(0).astype(int), adx.shift(1).fillna(0)

# 4小时版回测引擎（年化用252*6=1512 bars）
class Engine4h(TimeSeriesEngine):
    pass  # ATR止损逻辑相同，只是annualize不同

engine = Engine4h(atr_mult=1.0, use_atr_stop=True)

configs = [
    # (名称, fast, slow, sp, ma, adx_t, adx_w)
    ('原参数',        12, 26,  9, 200, 35, 20),
    ('适配4h×6',     72, 156, 54, 1200, 35, 120),
    ('中间参数',      36, 78, 27, 600, 35, 60),
    ('短周期',        6,  13,  4, 100, 30, 14),
    ('超短周期',      3,   8,  3,  50, 25, 10),
]

print()
print(f'{'配置':<12} {'全期Sh':>8} {'MaxDD':>8} {'年化':>8} {'2022Sh':>8} {'2024Sh':>8} {'持仓%':>7}')
print('-'*65)

for name, fast, slow, sp, ma_f, adx_t, adx_w in configs:
    try:
        sig, adx = signal_4h(ohlcv, fast, slow, sp, ma_f, adx_t, adx_w)
        r = engine.run(sig, ohlcv, start_date='2020-01-01')
        if not isinstance(r, pd.DataFrame) or r.empty:
            print(f'{name:<12} NO DATA')
            continue
        m = full_metrics(r['return'])
        r22 = r[r.index.year == 2022]
        r24 = r[r.index.year == 2024]
        m22 = full_metrics(r22['return']) if len(r22) > 20 else {}
        m24 = full_metrics(r24['return']) if len(r24) > 20 else {}
        hold = (r['position']==1).mean()
        print(f'{name:<12} {m["sharpe"]:+>8.3f} {m["max_drawdown"]:+>8.1%} {m["annual_return"]:+>8.1%} '
              f'{m22.get("sharpe",0):+>8.3f} {m24.get("sharpe",0):+>8.3f} {hold:>7.1%}')
    except Exception as e:
        print(f'{name:<12} ERROR: {e}')

# 额外：纯多头4h趋势（无ADX过滤），作为基准
print()
print('基准：纯MACD+MA200（无ADX过滤）')
sig_base = ohlcv['close'].copy()
ema_f = sig_base.ewm(span=12, adjust=False).mean()
ema_s = sig_base.ewm(span=26, adjust=False).mean()
hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=9, adjust=False).mean()
ma200 = sig_base.rolling(200).mean()
base_sig = ((hist > 0) & (sig_base > ma200)).astype(int).shift(1).fillna(0).astype(int)
r_base = engine.run(base_sig, ohlcv, start_date='2020-01-01')
if isinstance(r_base, pd.DataFrame) and not r_base.empty:
    m_base = full_metrics(r_base['return'])
    hold_base = (r_base['position']==1).mean()
    print(f'  Sharpe={m_base["sharpe"]:.3f} MaxDD={m_base["max_drawdown"]:.1%} Annual={m_base["annual_return"]:+.1%} Hold={hold_base:.1%}')
