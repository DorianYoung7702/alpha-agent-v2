"""
aggressive_v5.py - v2.0 策略优化版

优化目标：提升持仓时间占比（原v1.4仅13%）
核���改变：用 EMA20/60 金叉死叉替代 MACD + ADX 严格过滤

信号逻辑：
- EMA20 > EMA60（金叉持仓）
- BTC close > MA200（全局门控）
- ADX > 20（趋势确认，比原版35宽松）
- ATR 1.0x 止损（保留）
- 牛市确认：2.0x 仓位（保留）
- 熊市：资金费率套利（保留）
"""
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

STABLE_YIELD   = 0.05/365
LENDING_YIELD  = 0.08/365
FUNDING_THRESH = 0.00005
SYMS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','AVAXUSDT']

# EMA参数
EMA_FAST = 20
EMA_SLOW = 60
ADX_BASE = 35   # Expert批准参数
ATR_MULT = 0.8  # Expert最终选���：ATR=0.8（非端点，过拟合风险更低）

# 仓位上限（Expert指令 2026-03-15）
MAX_SINGLE_WEIGHT = 0.30  # ���币最大30%
MAX_TOTAL_WEIGHT  = 1.50  # 总仓位最大150%

t0 = time.time()
data    = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMBOLS_TOP30, use_cache=True)
panel   = build_panel(data, funding_data=funding, min_history_days=60)

btc             = data['BTCUSDT']
btc_close       = btc['close']
btc_ma200       = btc_close.rolling(200).mean()
btc_above_ma200 = (btc_close > btc_ma200).shift(1).fillna(False)
btc_adx_global  = calc_adx(btc, window=20).shift(1).fillna(0)
above_10d       = btc_above_ma200.rolling(10).min().fillna(0).astype(bool)
bull_market     = (above_10d & (btc_adx_global > 25)).shift(1).fillna(False)

# 资金费率
def get_funding_series(sym):
    try:
        return panel.xs(sym, level='symbol')['funding_rate'].shift(1).fillna(0)
    except Exception:
        return pd.Series(dtype=float)

funding_series = {sym: get_funding_series(sym) for sym in SYMS}

def get_bear_return(date):
    daily_rets = []
    for sym in SYMS:
        rate = funding_series[sym].loc[date] if date in funding_series[sym].index else 0.0
        daily_rets.append(abs(rate)*3 if abs(rate) >= FUNDING_THRESH else LENDING_YIELD)
    return float(np.mean(daily_rets)) if daily_rets else LENDING_YIELD

# ── EMA20/60 信号（新��� ───────────────────────────────────
def make_ema_sig(sym, adx_thresh=ADX_BASE, use_bull=True):
    ohlcv = data[sym]
    c     = ohlcv['close']
    ema_f = c.ewm(span=EMA_FAST, adjust=False).mean()
    ema_s = c.ewm(span=EMA_SLOW, adjust=False).mean()
    ma200 = c.rolling(200).mean()
    adx   = calc_adx(ohlcv, window=14)  # ADX窗口也放宽到14

    # 牛市时ADX门槛进一步降至15
    if use_bull:
        bull_r  = bull_market.reindex(adx.index).fillna(False)
        adx_thr = pd.Series(float(adx_thresh), index=adx.index)
        adx_thr[bull_r] = 15.0
    else:
        adx_thr = adx_thresh

    # EMA金叉 + MA200上方 + ADX确认
    sig = ((ema_f > ema_s) & (c > ma200) & (adx > adx_thr)).astype(int)
    sig = sig & btc_above_ma200.reindex(sig.index).fillna(False)
    return sig.shift(1).fillna(0).astype(int)

# ── BB均值回归（震荡市补充）──────���──────────────────────
def make_mr_sig(sym):
    c     = data[sym]['close']
    mid   = c.rolling(10).mean()
    lower = mid - 1.5 * c.rolling(10).std()
    pos   = pd.Series(0, index=c.index)
    in_t  = False
    for i in range(len(pos)):
        if not in_t and c.iloc[i] < lower.iloc[i]: in_t = True
        elif in_t and c.iloc[i] > mid.iloc[i]:     in_t = False
        pos.iloc[i] = 1 if in_t else 0
    pos = (pos.astype(bool) & btc_above_ma200.reindex(pos.index).fillna(False)).astype(int)
    return pos.shift(1).fillna(0).astype(int)

# ── 单币回��� ──────────────────────────────────────────────
def run_sym(sym, base_w=0.2, start=None):
    ema_sig = make_ema_sig(sym)
    mr_sig  = make_mr_sig(sym)
    engine_e = TimeSeriesEngine(atr_mult=ATR_MULT, use_atr_stop=True)
    engine_m = TimeSeriesEngine(atr_mult=ATR_MULT, use_atr_stop=True)
    r_ema = engine_e.run(ema_sig, data[sym], start_date=start)
    r_mr  = engine_m.run(mr_sig,  data[sym], start_date=start)
    if not isinstance(r_ema, pd.DataFrame) or r_ema.empty:
        return pd.Series(dtype=float)

    rets    = []
    cum_ret = 0.0
    for date, row in r_ema.iterrows():
        adx_val = btc_adx_global.loc[date] if date in btc_adx_global.index else 20
        is_bull = bull_market.loc[date]     if date in bull_market.index     else False
        btc_ok  = btc_above_ma200.loc[date] if date in btc_above_ma200.index else False
        w = base_w * (2.0 if is_bull else 1.0)

        if not btc_ok:
            # 熊市：资金费率套利
            rets.append(get_bear_return(date) * base_w)
            cum_ret = 0.0
            continue

        if row['position'] == 1:
            # 趋势持仓（动量加仓，单币上限30%）
            cum_ret += row['return']
            if cum_ret > 0.05:
                w = min(w * 1.5, MAX_SINGLE_WEIGHT)
            w = min(w, MAX_SINGLE_WEIGHT)  # 强制单币上限
            rets.append(row['return'] * w)
        elif adx_val < 20 and date in r_mr.index and r_mr.loc[date, 'position'] == 1:
            # 超低ADX（震荡市）：BB均值回归
            rets.append(r_mr.loc[date, 'return'] * base_w * 0.5)
            cum_ret = 0.0
        else:
            rets.append(STABLE_YIELD * base_w)
            cum_ret = 0.0

    return pd.Series(rets, index=r_ema.index)


def run_portfolio(start='2020-01-01'):
    all_rets = [run_sym(s, 0.2, start) for s in SYMS]
    valid = [r for r in all_rets if not r.empty]
    if not valid: return pd.Series(dtype=float)
    combined = pd.concat(valid, axis=1).fillna(0)
    # 总仓位压缩：如合计超过MAX_TOTAL_WEIGHT则等比压缩
    daily_total_w = combined.abs().sum(axis=1)
    scale = daily_total_w.apply(lambda x: min(1.0, MAX_TOTAL_WEIGHT / x) if x > MAX_TOTAL_WEIGHT else 1.0)
    combined = combined.multiply(scale, axis=0)
    return combined.sum(axis=1)


def report(ret, label):
    m   = full_metrics(ret)
    oos = ret[ret.index >= '2022-01-01']
    mo  = full_metrics(oos)
    targets = {2020:0.60, 2021:1.50, 2022:0.0, 2023:0.80, 2024:0.60, 2025:0.15}
    print(f'\n[{label}]')
    print(f'  全期: Sh={m["sharpe"]:.3f} DD={m["max_drawdown"]:.1%} Ann={m["annual_return"]:.1%}')
    print(f'  OOS:  Sh={mo["sharpe"]:.3f} DD={mo["max_drawdown"]:.1%} Ann={mo["annual_return"]:.1%}')

    # 持仓时间统计
    total_days = len(ret)
    bull_days  = btc_above_ma200.reindex(ret.index).fillna(False).sum()
    print(f'  BTC>MA200天数: {bull_days}/{total_days} ({bull_days/total_days*100:.1f}%)')

    for yr in range(2020, 2026):
        r = ret[ret.index.year == yr]
        if len(r) > 5:
            mm = full_metrics(r)
            t  = targets.get(yr, 0)
            f  = '✓' if mm['annual_return'] >= t else '✗'
            # 持仓占比
            btc_yr = btc_above_ma200.reindex(r.index).fillna(False)
            hold_pct = btc_yr.mean() * 100
            print(f'    {yr}: Ann={mm["annual_return"]:+.1%} Sh={mm["sharpe"]:+.3f} '
                  f'DD={mm["max_drawdown"]:.1%} 牛市占比={hold_pct:.0f}% {f}')
    return m, mo


if __name__ == '__main__':
    print(f'EMA{EMA_FAST}/{EMA_SLOW} + ADX{ADX_BASE} + ATR{ATR_MULT} 策略回测...')
    ret = run_portfolio('2020-01-01')
    m, mo = report(ret, f'v2.0 EMA{EMA_FAST}/{EMA_SLOW} 激进版')
    print(f'\nTotal: {time.time()-t0:.1f}s')
