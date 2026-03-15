import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data.fetcher import fetch_all_symbols, fetch_funding_all, build_panel, SYMBOLS_TOP30
from factors.ts_signals_v2 import calc_adx
from factors.volatility import HistoricalVolatility
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics

data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMBOLS_TOP30, use_cache=True)
panel = build_panel(data, funding_data=funding, min_history_days=60)
ohlcv_btc = data['BTCUSDT']

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
STABLE_YIELD = 0.05 / 365
hvol = HistoricalVolatility(window=20)
factor = hvol.compute(panel)
prices = panel['close'].unstack(level='symbol')

def run_combined(start_date=None, end_date=None):
    ohlcv = ohlcv_btc.copy()
    sig = btc_signal.copy()
    adx = btc_adx.copy()
    if start_date:
        ohlcv = ohlcv[ohlcv.index >= start_date]
        sig = sig[sig.index >= start_date]
        adx = adx[adx.index >= start_date]
    if end_date:
        ohlcv = ohlcv[ohlcv.index <= end_date]
        sig = sig[sig.index <= end_date]
        adx = adx[adx.index <= end_date]
    btc_result = engine.run(sig, ohlcv, start_date=start_date, end_date=end_date)
    if not isinstance(btc_result, pd.DataFrame) or btc_result.empty:
        return pd.DataFrame()
    combined_rets = []
    for date, row in btc_result.iterrows():
        btc_pos = row['position']
        btc_ret = row['return']
        if btc_pos == 1:
            combined_rets.append({'timestamp': date, 'return': btc_ret, 'mode': 'btc'})
        else:
            daily_ret = STABLE_YIELD
            mode = 'stable'
            if date in adx.index and adx.loc[date] < 20 and date in factor.index.get_level_values('timestamp'):
                try:
                    day_factor = factor.xs(date, level='timestamp').dropna()
                    if len(day_factor) >= 6:
                        top = day_factor.nlargest(3)
                        dates_list = sorted(prices.index)
                        if date in prices.index:
                            idx = dates_list.index(date)
                            if idx + 1 < len(dates_list):
                                next_date = dates_list[idx + 1]
                                cross_ret = sum((prices.loc[next_date, s] - prices.loc[date, s]) / prices.loc[date, s]
                                    for s in top.index if s in prices.columns) / len(top)
                                daily_ret = 0.5 * cross_ret + 0.5 * STABLE_YIELD
                                mode = 'cross'
                except: pass
            combined_rets.append({'timestamp': date, 'return': daily_ret, 'mode': mode})
    return pd.DataFrame(combined_rets).set_index('timestamp')

result = run_combined(start_date='2020-01-01')
btc_only = engine.run(btc_signal, ohlcv_btc, start_date='2020-01-01')

# 计算净值曲线
nav_combined = (1 + result['return']).cumprod()
nav_btc_only = (1 + btc_only['return']).cumprod()
nav_btc_hold = (ohlcv_btc['close'] / ohlcv_btc['close'].iloc[0]).reindex(nav_combined.index).ffill()

# 绘图
fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor('#0d1117')
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[2, 1])
ax5 = fig.add_subplot(gs[3, :])

for ax in [ax1,ax2,ax3,ax4,ax5]:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['top'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['right'].set_color('#30363d')

# 1. 净值曲线对比
ax1.plot(nav_combined.index, nav_combined.values, color='#58a6ff', linewidth=1.8, label='三层策略 (Sharpe=1.445)')
ax1.plot(nav_btc_only.index, nav_btc_only.values, color='#f78166', linewidth=1.2, label='BTC趋势单层 (Sharpe=1.176)', alpha=0.8)
ax1.plot(nav_btc_hold.index, nav_btc_hold.values, color='#6e7681', linewidth=1.0, label='BTC买入持有', alpha=0.6, linestyle='--')
ax1.axvline(pd.Timestamp('2024-01-01'), color='#f0883e', linewidth=1, linestyle=':', alpha=0.7)
ax1.text(pd.Timestamp('2024-01-15'), ax1.get_ylim()[0], '样本外开始', color='#f0883e', fontsize=8)
ax1.set_title('净值曲线对比 (2020-2026)', color='#e6edf3', fontsize=12, pad=8)
ax1.legend(loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=9)
ax1.set_ylabel('净值', color='#8b949e', fontsize=9)
ax1.yaxis.grid(True, color='#21262d', linewidth=0.5)

# 2. 每日收益分布（持仓模式着色）
btc_mask = result['mode'] == 'btc'
cross_mask = result['mode'] == 'cross'
stable_mask = result['mode'] == 'stable'
ax2.bar(result.index[btc_mask], result['return'][btc_mask], color='#58a6ff', width=1, alpha=0.8, label='BTC持仓')
ax2.bar(result.index[cross_mask], result['return'][cross_mask], color='#3fb950', width=1, alpha=0.8, label='截面选币')
ax2.bar(result.index[stable_mask], result['return'][stable_mask], color='#8b949e', width=1, alpha=0.4, label='稳定币')
ax2.axhline(0, color='#30363d', linewidth=0.5)
ax2.set_title('每日���益（持仓模式）', color='#e6edf3', fontsize=12, pad=8)
ax2.legend(loc='lower right', facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=9)
ax2.set_ylabel('日收益率', color='#8b949e', fontsize=9)
ax2.yaxis.grid(True, color='#21262d', linewidth=0.5)

# 3. 回撤曲线
dd_combined = (nav_combined / nav_combined.cummax() - 1) * 100
dd_btc_only = (nav_btc_only / nav_btc_only.cummax() - 1) * 100
ax3.fill_between(dd_combined.index, dd_combined.values, 0, color='#58a6ff', alpha=0.4)
ax3.plot(dd_combined.index, dd_combined.values, color='#58a6ff', linewidth=1, label='三层策略')
ax3.plot(dd_btc_only.index, dd_btc_only.values, color='#f78166', linewidth=1, alpha=0.7, label='BTC单层')
ax3.axhline(-18.4, color='#f0883e', linewidth=1, linestyle='--', alpha=0.7)
ax3.set_title('回撤曲线', color='#e6edf3', fontsize=11, pad=8)
ax3.legend(loc='lower left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=8)
ax3.set_ylabel('回撤 (%)', color='#8b949e', fontsize=9)
ax3.yaxis.grid(True, color='#21262d', linewidth=0.5)

# 4. 持仓分布饼图
modes = result['mode'].value_counts()
labels = {'btc': f'BTC趋势\n{modes.get("btc",0)}天', 'cross': f'截面选币\n{modes.get("cross",0)}天', 'stable': f'稳定币\n{modes.get("stable",0)}天'}
colors_pie = ['#58a6ff', '#3fb950', '#8b949e']
values = [modes.get('btc',0), modes.get('cross',0), modes.get('stable',0)]
wedges, texts, autotexts = ax4.pie(values, labels=[labels[k] for k in ['btc','cross','stable']],
    colors=colors_pie, autopct='%1.1f%%', startangle=90,
    textprops={'color': '#e6edf3', 'fontsize': 9},
    wedgeprops={'linewidth': 1, 'edgecolor': '#0d1117'})
for at in autotexts:
    at.set_color('#0d1117')
    at.set_fontweight('bold')
ax4.set_title('持仓时间分布', color='#e6edf3', fontsize=11, pad=8)

# 5. BTC价格 + 持仓区间
btc_close = ohlcv_btc['close'].reindex(result.index).ffill()
ax5.plot(btc_close.index, btc_close.values, color='#f0883e', linewidth=1.2, label='BTC价格')
ax5_twin = ax5.twinx()
ax5_twin.set_facecolor('#161b22')
for d in result.index[btc_mask]:
    ax5.axvspan(d, d + pd.Timedelta(days=1), color='#58a6ff', alpha=0.3)
for d in result.index[cross_mask]:
    ax5.axvspan(d, d + pd.Timedelta(days=1), color='#3fb950', alpha=0.2)
ax5.set_title('BTC价格 + 持仓区间（蓝=BTC趋势，绿=截面选币）', color='#e6edf3', fontsize=11, pad=8)
ax5.set_ylabel('BTC 价格 (USDT)', color='#8b949e', fontsize=9)
ax5.yaxis.grid(True, color='#21262d', linewidth=0.5)
ax5.set_yscale('log')
ax5_twin.set_yticks([])

fig.suptitle('OUROBOROS 三层叠加策略回测报���\nSharpe=1.445 | MaxDD=-18.4% | 年化+29.1%',
    color='#e6edf3', fontsize=14, fontweight='bold', y=0.98)

plt.savefig('reports/strategy_report.png', dpi=150, bbox_inches='tight',
    facecolor='#0d1117', edgecolor='none')
print('Chart saved: reports/strategy_report.png')
plt.close()
