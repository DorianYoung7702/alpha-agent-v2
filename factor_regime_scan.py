import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from scipy import stats
from data.fetcher import fetch_all_symbols, fetch_funding_all, build_panel, SYMBOLS_TOP30
from factors.volatility import HistoricalVolatility, DownsideVolatility
from factors.momentum import ReturnMomentum, RSIMomentum, FundingRateReversal
from factors.volume import MoneyFlow, VWAP_Deviation
from factors.ts_signals_v2 import calc_adx

data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMBOLS_TOP30, use_cache=True)
panel = build_panel(data, funding_data=funding, min_history_days=60)
ohlcv_btc = data['BTCUSDT']
btc_adx = calc_adx(ohlcv_btc, window=20).shift(1).fillna(0)

# 只取 ADX<20 的日期
low_adx_dates = set(btc_adx[btc_adx < 20].index)

close = panel['close'].unstack(level='symbol')
fwd_ret = (close.shift(-1)/close - 1).stack()
fwd_ret.index.names = ['timestamp', 'symbol']

def calc_ic_regime(factor_series, dates_filter):
    combined = pd.concat([factor_series.rename('f'), fwd_ret.rename('r')], axis=1).dropna()
    combined_filtered = combined[combined.index.get_level_values('timestamp').isin(dates_filter)]
    if len(combined_filtered) < 50:
        return 0.0, 0.0, 0
    def ic_row(g):
        if len(g) < 5: return np.nan
        c, _ = stats.spearmanr(g['f'], g['r'])
        return c
    ic = combined_filtered.groupby(level='timestamp').apply(ic_row).dropna()
    return ic.mean(), ic.mean()/(ic.std()+1e-10), len(ic)

factors_to_test = [
    ('hvol_20d', HistoricalVolatility(window=20)),
    ('hvol_10d', HistoricalVolatility(window=10)),
    ('downvol_20d', DownsideVolatility(window=20)),
    ('mom_5d', ReturnMomentum(window=5)),
    ('mom_10d', ReturnMomentum(window=10)),
    ('mom_21d', ReturnMomentum(window=21)),
    ('rsi_14', RSIMomentum(window=14)),
    ('rsi_21', RSIMomentum(window=21)),
    ('money_flow_5d', MoneyFlow(window=5)),
    ('money_flow_10d', MoneyFlow(window=10)),
    ('vwap_dev_5d', VWAP_Deviation(window=5)),
    ('vwap_dev_10d', VWAP_Deviation(window=10)),
]

# 尝试加入资金费率因子
try:
    factors_to_test.append(('fr_reversal_7d', FundingRateReversal(window=7)))
except: pass

print(f'ADX<20 天数: {len(low_adx_dates)}')
print(f'{'因子':<20} {'IC(ADX<20)':>12} {'ICIR':>8} {'天数':>6} {'IC(全���)':>10}')
print('-'*60)

results = []
for name, factor_obj in factors_to_test:
    try:
        f = factor_obj.compute(panel)
        ic_low, icir_low, n = calc_ic_regime(f, low_adx_dates)
        # 全期IC
        combined_all = pd.concat([f.rename('f'), fwd_ret.rename('r')], axis=1).dropna()
        ic_all_series = combined_all.groupby(level='timestamp').apply(
            lambda g: stats.spearmanr(g['f'], g['r'])[0] if len(g)>=5 else np.nan
        ).dropna()
        ic_all = ic_all_series.mean()
        print(f'{name:<20} {ic_low:>12.4f} {icir_low:>8.4f} {n:>6} {ic_all:>10.4f}')
        results.append((name, ic_low, icir_low, n, ic_all))
    except Exception as e:
        print(f'{name:<20} ERROR: {e}')

results.sort(key=lambda x: x[1], reverse=True)
print()
print('ADX<20 最佳因子 Top5:')
for r in results[:5]:
    print(f'  {r[0]}: IC={r[1]:.4f}, ICIR={r[2]:.4f}')
