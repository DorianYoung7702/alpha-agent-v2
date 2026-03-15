import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from data.fetcher import fetch_klines
from factors.ts_signals_v2 import calc_adx
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics

def signal_macd_ma_adx(ohlcv, fast=12, slow=26, sp=9, ma_f=200, adx_t=35, adx_w=20):
    close = ohlcv['close']
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=sp, adjust=False).mean()
    ma = close.rolling(ma_f).mean()
    adx = calc_adx(ohlcv, window=adx_w)
    result = ((hist > 0) & (close > ma) & (adx > adx_t)).astype(int)
    return result.shift(1).fillna(0).astype(int)

# 对每个品种独立搜索最优参数
symbols_configs = {
    'ETHUSDT': [
        (12, 26, 9, 200, 35, 20, 1.0),
        (12, 26, 9, 200, 30, 20, 1.0),
        (12, 26, 9, 200, 40, 20, 1.0),
        (12, 26, 9, 150, 35, 20, 1.0),
        (12, 26, 9, 200, 35, 14, 1.0),
        (10, 24, 7, 200, 35, 20, 1.0),
        (12, 26, 9, 200, 35, 20, 1.5),
    ],
    'SOLUSDT': [
        (12, 26, 9, 200, 35, 20, 1.0),
        (12, 26, 9, 200, 30, 20, 1.0),
        (12, 26, 9, 200, 40, 20, 1.0),
        (12, 26, 9, 100, 35, 20, 1.0),
        (12, 26, 9, 200, 35, 14, 1.0),
        (10, 24, 7, 150, 35, 20, 1.0),
        (12, 26, 9, 200, 35, 20, 1.5),
    ],
    'BNBUSDT': [
        (12, 26, 9, 200, 35, 20, 1.0),
        (12, 26, 9, 200, 30, 20, 1.0),
        (12, 26, 9, 150, 35, 20, 1.0),
    ],
}

STABLE_YIELD = 0.05 / 365

for sym, configs in symbols_configs.items():
    print(f'\n{"="*60}')
    print(f'{sym}')
    print(f'{"="*60}')
    ohlcv = fetch_klines(sym, '1d', '2020-01-01', use_cache=True)
    print(f'Data: {len(ohlcv)} days ({ohlcv.index.min().date()} ~ {ohlcv.index.max().date()})')
    
    results = []
    for fast, slow, sp, ma_f, adx_t, adx_w, atr_m in configs:
        try:
            engine = TimeSeriesEngine(atr_mult=atr_m, use_atr_stop=True)
            sig = signal_macd_ma_adx(ohlcv, fast, slow, sp, ma_f, adx_t, adx_w)
            
            # 加稳定币收益（空仓期）
            r = engine.run(sig, ohlcv, start_date='2020-01-01')
            if not isinstance(r, pd.DataFrame) or r.empty:
                continue
            
            # 叠加���定币收益
            r['return'] = r.apply(lambda x: x['return'] if x['position']==1 else STABLE_YIELD, axis=1)
            
            m = full_metrics(r['return'])
            r22 = r[r.index.year == 2022]
            r_oos = r[r.index >= '2022-01-01']
            m22 = full_metrics(r22['return']) if len(r22)>20 else {}
            m_oos = full_metrics(r_oos['return']) if len(r_oos)>20 else {}
            hold = (r['position']==1).mean()
            
            results.append((
                f'macd/ma{ma_f}/adx{adx_t}w{adx_w}/atr{atr_m}',
                m['sharpe'], m['max_drawdown'], m['annual_return'],
                m22.get('sharpe',0), m_oos.get('sharpe',0), m_oos.get('max_drawdown',-1), hold
            ))
        except Exception as e:
            pass
    
    results.sort(key=lambda x: x[1], reverse=True)
    print(f'{"配置":<35} {"全Sh":>6} {"MaxDD":>7} {"年化":>7} {"22Sh":>6} {"OOSSh":>7} {"OOSDD":>7} {"持仓":>6}')
    print('-'*85)
    for r in results:
        print(f'{r[0]:<35} {r[1]:+>6.3f} {r[2]:+>7.1%} {r[3]:+>7.1%} {r[4]:+>6.3f} {r[5]:+>7.3f} {r[6]:+>7.1%} {r[7]:>6.1%}')
