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

# 修复记录（2026-03-15）：
# - 问题2/3/14：引擎已改为���当日信号，信号函数内shift(1)保留，消除双重延迟
# - 问题5：删除Layer2/3前视偏差逻辑
# - 问题7：引擎已改为次日开盘平仓
# - 问题13：趋势优先于MR，ADX 20-25区间以趋势信号为准（position==1检���在前）
# - 注：牛市仓位2x为设计意图（隐性杠杆），未计融资成本，汇报时需说明

STABLE_YIELD = 0.05/365
SYMS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','AVAXUSDT']

t0 = time.time()
data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMBOLS_TOP30, use_cache=True)
panel = build_panel(data, funding_data=funding, min_history_days=60)

btc_close = data['BTCUSDT']['close']
btc_ma200 = btc_close.rolling(200).mean()
btc_above_ma200 = (btc_close > btc_ma200).shift(1).fillna(False)
btc_adx_global = calc_adx(data['BTCUSDT'], window=20).shift(1).fillna(0)

# 修复问题3：make_bull_market 只用一次shift（btc_above_ma200已shift(1)，此处不再shift）
def make_bull_market(confirm_days=10):
    above_nd = btc_above_ma200.rolling(confirm_days).min().fillna(0).astype(bool)
    adx_ok = (btc_adx_global > 25)
    # 不再额外shift(1)，btc_above_ma200已经shift过了
    return (above_nd & adx_ok).fillna(False)

# 修复问题14：btc_ma200_breakout 只用一次shift
btc_ma200_breakout = (
    btc_above_ma200 &  # ���shift(1)
    ~btc_above_ma200.shift(5).fillna(True) &
    ((btc_close.shift(1) - btc_close.shift(6)) / btc_close.shift(6) > 0.05)  # 用shift后的价格
).fillna(False)  # 不再额外shift(1)

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
    # 问题13注释：ADX 20-25区间趋势和MR均可触发，趋势优先（run_sym中position==1判断在前）
    sig = ((hist>0)&(c>ma200)&(adx>adx_thresh_s)).astype(int)
    sig = sig & btc_above_ma200.reindex(sig.index).fillna(False)
    return sig.shift(1).fillna(0).astype(int)  # 信号函数内shift(1)

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
    return pos.shift(1).fillna(0).astype(int)  # 信号函数内shift(1)

def run_sym(sym, base_w=0.2, start=None, bull_market=None,
           bull_mult=2.0, momentum_thresh=0.05, momentum_add=0.5):
    ohlcv = data[sym]
    trend_sig = make_trend_sig(ohlcv, bull_market)
    mr_sig = make_mr_sig(ohlcv)
    engine_t = TimeSeriesEngine(atr_mult=1.0, use_atr_stop=True)
    engine_m = TimeSeriesEngine(atr_mult=1.0, use_atr_stop=True)
    r_trend = engine_t.run(trend_sig, ohlcv, start_date=start)
    r_mr = engine_m.run(mr_sig, ohlcv, start_date=start)
    if not isinstance(r_trend, pd.DataFrame) or r_trend.empty:
        return pd.Series(dtype=float)
    rets = []
    cum_ret = 0.0
    for date, row in r_trend.iterrows():
        adx_val = btc_adx_global.loc[date] if date in btc_adx_global.index else 25
        is_bull = bull_market.loc[date] if date in bull_market.index else False
        btc_ok = btc_above_ma200.loc[date] if date in btc_above_ma200.index else False
        w = base_w * (bull_mult if is_bull else 1.0)
        if row['position'] == 1:
            cum_ret += row['return']
            if cum_ret > momentum_thresh:
                w = min(w*(1+momentum_add), base_w*(bull_mult+momentum_add))
        else:
            cum_ret = 0.0
        if not btc_ok:
            rets.append(STABLE_YIELD*base_w)
            continue
        # 趋势优先（问题13）：position==1检查在MR之前
        if row['position'] == 1:
            rets.append(row['return']*w)
        elif adx_val < 25 and date in r_mr.index and r_mr.loc[date,'position'] == 1:
            # MR模块（ADX<25震荡市，趋势不活跃时才启用）
            rets.append(r_mr.loc[date,'return']*base_w*0.5)
        else:
            # 问题5修复：删除Layer2/3前视偏差，改为纯稳定币
            rets.append(STABLE_YIELD*base_w)
    return pd.Series(rets, index=r_trend.index)

def run_portfolio(start=None, bull_mult=2.0, momentum_thresh=0.05, momentum_add=0.5):
    bm = make_bull_market(10)
    all_rets = [run_sym(s,0.2,start,bm,bull_mult,momentum_thresh,momentum_add) for s in SYMS]
    return pd.concat([r for r in all_rets if not r.empty],axis=1).fillna(0).sum(axis=1)

def report(ret, label):
    m = full_metrics(ret)
    oos = ret[ret.index>='2022-01-01']
    mo = full_metrics(oos)
    targets = {2020:0.60,2021:1.50,2022:0.0,2023:0.80,2024:0.60,2025:0.15}
    print('\n[%s]' % label)
    print('  全期: Sh=%.3f DD=%.1f%% Ann=%.1f%%'%(m['sharpe'],m['max_drawdown']*100,m['annual_return']*100))
    print('  OOS:  Sh=%.3f DD=%.1f%% Ann=%.1f%%'%(mo['sharpe'],mo['max_drawdown']*100,mo['annual_return']*100))
    for yr in range(2020,2026):
        r = ret[ret.index.year==yr]
        if len(r)>5:
            mm = full_metrics(r)
            t = targets.get(yr,0)
            f = '✓' if mm['annual_return']>=t else '✗'
            print('    %d: Ann=%+.1f%% Sh=%+.3f DD=%.1f%% %s'%(
                yr,mm['annual_return']*100,mm['sharpe'],mm['max_drawdown']*100,f))
    return m, mo

if __name__ == '__main__':
    print('Loading...')
    ret = run_portfolio(start='2020-01-01')
    report(ret, 'v1.4 激进版（修复：双重shift+Layer2/3+平仓时点）')
    print('\nTotal: %.1fs'%(time.time()-t0))
