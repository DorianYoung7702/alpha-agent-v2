import sys
sys.path.insert(0, r'D:\YZX\alpha-agent\src')
from loguru import logger; logger.remove(); logger.add(sys.stdout, level='WARNING')
import pandas as pd
import numpy as np
from data.fetcher import fetch_all_symbols, fetch_funding_all, build_panel, SYMBOLS_TOP30
from factors.ts_signals_v2 import calc_adx
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics
import time

# v1.4_bear.py - 激进版 + 熊市费率套利模块
# 新增：BTC < MA200 期间动态费率套利（与稳健版相同逻辑）
# 本金：$100,000

INITIAL_CAPITAL = 100_000
STABLE_YIELD  = 0.05 / 365
LENDING_YIELD = 0.08 / 365
FUNDING_THRESH = 0.00005
SYMS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','AVAXUSDT']

t0 = time.time()
print('Loading data...')
data    = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMBOLS_TOP30, use_cache=True)
panel   = build_panel(data, funding_data=funding, min_history_days=60)

btc_close       = data['BTCUSDT']['close']
btc_ma200       = btc_close.rolling(200).mean()
btc_above_ma200 = (btc_close > btc_ma200).shift(1).fillna(False)
btc_adx_global  = calc_adx(data['BTCUSDT'], window=20).shift(1).fillna(0)

# 资金���率序列
def get_funding_series(sym):
    try:
        return panel.xs(sym, level='symbol')['funding_rate'].shift(1).fillna(0)
    except:
        return pd.Series(dtype=float)
funding_series = {sym: get_funding_series(sym) for sym in SYMS}

def get_bear_return(date):
    daily_rets = []
    for sym in SYMS:
        fr = funding_series[sym]
        rate = fr.loc[date] if date in fr.index else 0.0
        if abs(rate) < FUNDING_THRESH:
            daily_rets.append(LENDING_YIELD)
        else:
            daily_rets.append(abs(rate) * 3)
    return float(np.mean(daily_rets)) if daily_rets else LENDING_YIELD

# 牛市确认（修复单重shift）
def make_bull_market(confirm_days=10):
    above_nd = btc_above_ma200.rolling(confirm_days).min().fillna(0).astype(bool)
    adx_ok   = (btc_adx_global > 25)
    return (above_nd & adx_ok).fillna(False)

# MA200突破加速信号（修复单重shift）
btc_ma200_breakout = (
    btc_above_ma200 &
    ~btc_above_ma200.shift(5).fillna(True) &
    ((btc_close.shift(1) - btc_close.shift(6)) / btc_close.shift(6) > 0.05)
).fillna(False)

def calc_bband(close, window=10, num_std=1.5):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return ma+num_std*std, ma, ma-num_std*std

def make_trend_sig(ohlcv, bull_market, adx_base=35):
    c = ohlcv['close']
    ef = c.ewm(span=12,adjust=False).mean()
    es = c.ewm(span=26,adjust=False).mean()
    hist = (ef-es)-(ef-es).ewm(span=9,adjust=False).mean()
    ma200 = c.rolling(200).mean()
    adx = calc_adx(ohlcv, window=20)
    bull_r = bull_market.reindex(adx.index).fillna(False)
    breakout_r = btc_ma200_breakout.reindex(adx.index).fillna(False)
    adx_thresh_s = pd.Series(float(adx_base), index=adx.index)
    adx_thresh_s[bull_r | breakout_r] = 20.0
    sig = ((hist>0)&(c>ma200)&(adx>adx_thresh_s)).astype(int)
    sig = sig & btc_above_ma200.reindex(sig.index).fillna(False)
    return sig.shift(1).fillna(0).astype(int)

def make_mr_sig(ohlcv):
    c = ohlcv['close']
    upper,mid,lower = calc_bband(c,10,1.5)
    pos = pd.Series(0,index=c.index)
    in_t = False
    for i in range(len(pos)):
        if not in_t and c.iloc[i]<lower.iloc[i]: in_t=True
        elif in_t and c.iloc[i]>mid.iloc[i]: in_t=False
        pos.iloc[i] = 1 if in_t else 0
    pos = (pos.astype(bool) & btc_above_ma200.reindex(pos.index).fillna(False)).astype(int)
    return pos.shift(1).fillna(0).astype(int)

def run_sym(sym, base_w=0.2, start=None, bull_market=None,
           bull_mult=2.0, momentum_thresh=0.05, momentum_add=0.5):
    ohlcv = data[sym]
    trend_sig = make_trend_sig(ohlcv, bull_market)
    mr_sig = make_mr_sig(ohlcv)
    engine_t = TimeSeriesEngine(atr_mult=1.0, use_atr_stop=True)
    engine_m = TimeSeriesEngine(atr_mult=1.0, use_atr_stop=True)
    r_trend = engine_t.run(trend_sig, ohlcv, start_date=start)
    r_mr    = engine_m.run(mr_sig,    ohlcv, start_date=start)
    if not isinstance(r_trend, pd.DataFrame) or r_trend.empty:
        return pd.Series(dtype=float)
    rets = []
    cum_ret = 0.0
    for date, row in r_trend.iterrows():
        adx_val = btc_adx_global.loc[date] if date in btc_adx_global.index else 25
        is_bull = bull_market.loc[date] if date in bull_market.index else False
        btc_ok  = btc_above_ma200.loc[date] if date in btc_above_ma200.index else False
        w = base_w * (bull_mult if is_bull else 1.0)
        if row['position'] == 1:
            cum_ret += row['return']
            if cum_ret > momentum_thresh:
                w = min(w*(1+momentum_add), base_w*(bull_mult+momentum_add))
        else:
            cum_ret = 0.0
        if not btc_ok:
            # 熊市���动态费率套利（新增）
            rets.append(get_bear_return(date) * base_w)
            continue
        if row['position'] == 1:
            rets.append(row['return'] * w)
        elif adx_val < 25 and date in r_mr.index and r_mr.loc[date,'position'] == 1:
            rets.append(r_mr.loc[date,'return'] * base_w * 0.5)
        else:
            rets.append(STABLE_YIELD * base_w)
    return pd.Series(rets, index=r_trend.index)

def run_portfolio(start=None, bull_mult=2.0, momentum_thresh=0.05, momentum_add=0.5):
    bm = make_bull_market(10)
    all_rets = [run_sym(s,0.2,start,bm,bull_mult,momentum_thresh,momentum_add) for s in SYMS]
    return pd.concat([r for r in all_rets if not r.empty],axis=1).fillna(0).sum(axis=1)

def report_capital(ret, label, capital=100_000):
    m   = full_metrics(ret)
    oos = ret[ret.index >= '2022-01-01']
    mo  = full_metrics(oos)
    cum = (1 + ret).cumprod()
    final_capital = capital * cum.iloc[-1]

    print('\n' + '='*60)
    print(label)
    print('='*60)
    print('初始本金：$%s' % f'{capital:,.0f}')
    print('最终资产：$%s' % f'{final_capital:,.0f}')
    print('总收益率：%.1f%%' % ((cum.iloc[-1]-1)*100))
    print()
    print('全期 Sharpe=%.3f  MaxDD=%.1f%%  年化=%.1f%%' % (
        m['sharpe'], m['max_drawdown']*100, m['annual_return']*100))
    print('OOS  Sharpe=%.3f  MaxDD=%.1f%%  年化=%.1f%%' % (
        mo['sharpe'], mo['max_drawdown']*100, mo['annual_return']*100))
    print()
    print('%-6s %12s %12s %10s %10s' % ('年份','年末资产($)','年化收益','Sharpe','MaxDD'))
    print('-'*55)
    running_cap = capital
    for yr in range(2020, 2026):
        r = ret[ret.index.year == yr]
        if len(r) > 5:
            mm  = full_metrics(r)
            yr_ret = (1+r).prod() - 1
            running_cap *= (1 + yr_ret)
            tag = '[OOS]' if yr >= 2022 else '     '
            print('%s%d  $%12s  %10.1f%%  %8.3f  %8.1f%%' % (
                tag, yr,
                f'{running_cap:,.0f}',
                mm['annual_return']*100,
                mm['sharpe'],
                mm['max_drawdown']*100))
    print()
    # 最大回撤时期
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd_date = dd.idxmin()
    print('最大回撤发生：%s（%.1f%%）' % (max_dd_date.strftime('%Y-%m-%d'), dd.min()*100))
    return m, mo

if __name__ == '__main__':
    print('Loading...')
    ret = run_portfolio(start='2020-01-01')
    report_capital(ret, 'v1.4_bear 激进版+熊市费率套利  |  初始本金 $100,000', INITIAL_CAPITAL)
    print('\nTotal: %.1fs' % (time.time()-t0))
