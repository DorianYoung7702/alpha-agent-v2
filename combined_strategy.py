import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from data.fetcher import fetch_all_symbols, fetch_funding_all, build_panel, SYMBOLS_TOP30
from factors.ts_signals_v2 import calc_adx
from factors.volatility import HistoricalVolatility
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics, print_metrics

# 加载���据
data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMBOLS_TOP30, use_cache=True)
panel = build_panel(data, funding_data=funding, min_history_days=60)
ohlcv_btc = data['BTCUSDT']

# === 基础信号 ===
def signal_macd_ma_adx(ohlcv, fast=12, slow=26, sp=9, ma_f=200, adx_t=35, adx_w=20):
    close = ohlcv['close']
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=sp, adjust=False).mean()
    ma = close.rolling(ma_f).mean()
    adx = calc_adx(ohlcv, window=adx_w)
    result = ((hist > 0) & (close > ma) & (adx > adx_t)).astype(int)
    return result.shift(1).fillna(0).astype(int), adx.shift(1).fillna(0)

btc_signal, btc_adx = signal_macd_ma_adx(ohlcv_btc)
engine = TimeSeriesEngine(atr_mult=1.0, use_atr_stop=True)

# === 截面因子（hvol_20d）===
hvol = HistoricalVolatility(window=20)
factor = hvol.compute(panel)

# === 组合回测 ===
STABLE_YIELD = 0.05 / 365  # 稳定币日收益
CROSS_TOP_N = 3  # 截面选币时选Top3

def run_combined(
    ohlcv_btc, btc_signal, btc_adx, factor, panel,
    start_date=None, end_date=None,
    use_stable=True, use_cross=True, adx_cross_max=20
):
    prices = panel['close'].unstack(level='symbol')
    
    if start_date:
        ohlcv_btc = ohlcv_btc[ohlcv_btc.index >= start_date]
        btc_signal = btc_signal[btc_signal.index >= start_date]
        btc_adx = btc_adx[btc_adx.index >= start_date]
    if end_date:
        ohlcv_btc = ohlcv_btc[ohlcv_btc.index <= end_date]
        btc_signal = btc_signal[btc_signal.index <= end_date]
        btc_adx = btc_adx[btc_adx.index <= end_date]
    
    # BTC回测
    btc_result = engine.run(btc_signal, ohlcv_btc,
        start_date=start_date, end_date=end_date)
    
    if not isinstance(btc_result, pd.DataFrame) or btc_result.empty:
        return pd.Series(dtype=float)
    
    combined_rets = []
    
    for date, row in btc_result.iterrows():
        btc_pos = row['position']
        btc_ret = row['return']
        
        if btc_pos == 1:
            # BTC持仓期：全仓BTC
            combined_rets.append({'timestamp': date, 'return': btc_ret, 'mode': 'btc'})
        else:
            # BTC空仓期
            daily_ret = 0.0
            mode = 'cash'
            
            # 层2：稳定币收益
            if use_stable:
                daily_ret += STABLE_YIELD
                mode = 'stable'
            
            # 层3：截面选币（ADX<20时才启用，避免熊市）
            if use_cross and date in btc_adx.index:
                adx_val = btc_adx.loc[date]
                if adx_val < 20 and date in factor.index.get_level_values('timestamp'):
                    try:
                        day_factor = factor.xs(date, level='timestamp').dropna()
                        if len(day_factor) >= CROSS_TOP_N * 2:
                            top = day_factor.nlargest(CROSS_TOP_N)
                            # 计���次日收益
                            if date in prices.index:
                                dates_list = sorted(prices.index)
                                idx = dates_list.index(date)
                                if idx + 1 < len(dates_list):
                                    next_date = dates_list[idx + 1]
                                    cross_ret = 0.0
                                    for sym in top.index:
                                        if sym in prices.columns:
                                            p0 = prices.loc[date, sym]
                                            p1 = prices.loc[next_date, sym]
                                            if p0 > 0:
                                                cross_ret += (p1 - p0) / p0
                                    cross_ret /= len(top)
                                    # 空仓期截面仓位50%，稳定币50%
                                    daily_ret = 0.5 * cross_ret + 0.5 * STABLE_YIELD
                                    mode = 'cross'
                    except Exception:
                        pass
            
            combined_rets.append({'timestamp': date, 'return': daily_ret, 'mode': mode})
    
    df = pd.DataFrame(combined_rets).set_index('timestamp')
    return df

# 测试各组合
configs = [
    ('BTC-only',        False, False),
    ('BTC+Stable',      True,  False),
    ('BTC+Stable+Cross',True,  True),
]

print('='*70)
print('组合策略对比（全期 2020-2026）')
print('='*70)
for name, use_s, use_c in configs:
    result = run_combined(ohlcv_btc, btc_signal, btc_adx, factor, panel,
        use_stable=use_s, use_cross=use_c)
    if isinstance(result, pd.DataFrame) and not result.empty:
        m = full_metrics(result['return'], factor_name=name)
        print(f'{name:25}: Sharpe={m["sharpe"]:+.3f} MaxDD={m["max_drawdown"]:.1%} Annual={m["annual_return"]:+.1%}')
        if use_c:
            modes = result['mode'].value_counts()
            print(f'  持仓分布: {dict(modes)}')

print()
print('样本外对比（2022-2026）')
print('-'*70)
for name, use_s, use_c in configs:
    result = run_combined(ohlcv_btc, btc_signal, btc_adx, factor, panel,
        start_date='2022-01-01', use_stable=use_s, use_cross=use_c)
    if isinstance(result, pd.DataFrame) and not result.empty:
        m = full_metrics(result['return'], factor_name=name)
        print(f'{name:25}: Sharpe={m["sharpe"]:+.3f} MaxDD={m["max_drawdown"]:.1%} Annual={m["annual_return"]:+.1%}')
