import sys; sys.path.insert(0, 'src')
import pandas as pd, numpy as np
from scipy import stats
from data.fetcher import fetch_all_symbols, fetch_funding_all, build_panel, SYMBOLS_TOP30
from factors.volatility import HistoricalVolatility
from factors.ts_signals_v2 import calc_adx

print('Loading data...')
data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMBOLS_TOP30, use_cache=True)
panel = build_panel(data, funding_data=funding, min_history_days=60)
ohlcv_btc = data['BTCUSDT']

print('Computing factors...')
hvol = HistoricalVolatility(window=20)
factor = hvol.compute(panel)
close = panel['close'].unstack(level='symbol')
fwd_ret = (close.shift(-1)/close - 1).stack()
fwd_ret.index.names = ['timestamp', 'symbol']
btc_adx = calc_adx(ohlcv_btc, window=20).shift(1).fillna(0)

print('Computing IC by ADX regime...')
combined = pd.concat([factor.rename('f'), fwd_ret.rename('r')], axis=1).dropna()
def ic_row(g):
    if len(g) < 5: return np.nan
    c, _ = stats.spearmanr(g['f'], g['r'])
    return c
ic = combined.groupby(level='timestamp').apply(ic_row).dropna()

low  = ic[ic.index.isin(btc_adx[btc_adx < 20].index)]
mid  = ic[ic.index.isin(btc_adx[(btc_adx >= 20) & (btc_adx < 35)].index)]
high = ic[ic.index.isin(btc_adx[btc_adx >= 35].index)]

print('\n' + '='*55)
print('hvol_20d IC 按 BTC ADX 分层')
print('='*55)
print(f'ADX < 20  (震荡/熊市): IC={low.mean():.4f}  ICIR={low.mean()/(low.std()+1e-10):.4f}  n={len(low)}')
print(f'ADX 20-35 (中等趋势): IC={mid.mean():.4f}  ICIR={mid.mean()/(mid.std()+1e-10):.4f}  n={len(mid)}')
print(f'ADX > 35  (强趋势):   IC={high.mean():.4f}  ICIR={high.mean()/(high.std()+1e-10):.4f}  n={len(high)}')
print('='*55)

# 判断
if low.mean() > 0.05:
    print('\n[结论] ADX<20 期间 IC>0.05 ✓ 截面选币逻辑成立，层3可用')
elif low.mean() > 0.03:
    print('\n[结论] ADX<20 期间 IC 在 0.03-0.05，信号偏弱，层3谨慎使用')
elif low.mean() > 0:
    print('\n[结论] ADX<20 期间 IC<0.03，���面选币无显著 Alpha，层3改为纯稳定币')
else:
    print('\n[结论] ADX<20 期间 IC<0 ！截面选���反向，层3直接删除')
