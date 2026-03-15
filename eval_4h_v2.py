import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from data.fetcher import fetch_klines
from factors.ts_signals_v2 import calc_adx
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics

ohlcv = fetch_klines('BTCUSDT', '4h', '2020-01-01', use_cache=True)
print(f'BTC 4h: {len(ohlcv)} bars')

def signal_4h_v2(ohlcv, fast, slow, sp, ma_f, adx_t, adx_w):
    close = ohlcv['close']
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=sp, adjust=False).mean()
    ma = close.rolling(ma_f).mean()
    adx = calc_adx(ohlcv, window=adx_w)
    result = ((hist > 0) & (close > ma) & (adx > adx_t)).astype(int)
    return result.shift(1).fillna(0).astype(int), adx.shift(1).fillna(0)

# ���对4h专门设计的参数网格
# 4h线每天6根K线，周期需要更短
configs = [
    # (fast, slow, sp, ma, adx_t, adx_w) - 专为4h设计
    (8,  21,  5,  120, 25, 10),   # 约等于日线2/3.5/0.8，短趋势
    (12, 26,  9,  200, 25, 14),   # 日线原参数，ADX降低
    (6,  14,  4,   72, 20, 10),   # 超短趋势
    (20, 50, 15,  300, 30, 20),   # 中等���势
    (30, 60, 20,  500, 35, 20),   # 较长趋势
    (8,  21,  5,  200, 30, 14),   # 中短趋势+长MA
    (12, 26,  9,  120, 30, 14),   # 原MACD+中MA
    (8,  21,  5,  120, 20, 10),   # 宽松ADX
    (15, 35, 10,  250, 30, 14),   # 新配置
    (10, 24,  7,  150, 28, 12),   # 折中配置
]

atr_mults = [1.5, 2.0, 2.5]

results = []
for fast, slow, sp, ma_f, adx_t, adx_w in configs:
    for atr_m in atr_mults:
        try:
            engine = TimeSeriesEngine(atr_mult=atr_m, use_atr_stop=True)
            sig, adx = signal_4h_v2(ohlcv, fast, slow, sp, ma_f, adx_t, adx_w)
            r = engine.run(sig, ohlcv, start_date='2020-01-01')
            if not isinstance(r, pd.DataFrame) or r.empty:
                continue
            m = full_metrics(r['return'])
            r22 = r[r.index.year == 2022]
            r24 = r[r.index.year == 2024]
            m22 = full_metrics(r22['return']) if len(r22) > 50 else {}
            m24 = full_metrics(r24['return']) if len(r24) > 50 else {}
            r_oos = r[r.index >= '2022-01-01']
            m_oos = full_metrics(r_oos['return']) if len(r_oos) > 50 else {}
            hold = (r['position']==1).mean()
            results.append((
                f'{fast}/{slow}/{sp}/ma{ma_f}/adx{adx_t}w{adx_w}/atr{atr_m}',
                m['sharpe'], m['max_drawdown'], m['annual_return'],
                m22.get('sharpe',0), m24.get('sharpe',0),
                m_oos.get('sharpe',0), m_oos.get('max_drawdown',-1),
                hold
            ))
        except Exception as e:
            pass

results.sort(key=lambda x: x[1], reverse=True)
print(f'\n{"参数":<45} {"全Sh":>6} {"MaxDD":>7} {"年化":>7} {"22Sh":>6} {"24Sh":>6} {"OOSSh":>7} {"OOSDD":>7} {"持仓":>6}')
print('-'*105)
for r in results[:20]:
    print(f'{r[0]:<45} {r[1]:+>6.3f} {r[2]:+>7.1%} {r[3]:+>7.1%} {r[4]:+>6.3f} {r[5]:+>6.3f} {r[6]:+>7.3f} {r[7]:+>7.1%} {r[8]:>6.1%}')

if results:
    print(f'\n总计: {len(results)} 个配置测试完成')
    passed = [(r) for r in results if r[1]>1.5 and r[2]>-0.40 and r[6]>0.8]
    print(f'通过标准(Sh>1.5, DD<40%, OOS>0.8): {len(passed)} 个')
    if passed:
        print('通过的配置:')
        for r in passed:
            print(f'  {r[0]}: Sh={r[1]:.3f} DD={r[2]:.1%} OOS={r[6]:.3f}')
