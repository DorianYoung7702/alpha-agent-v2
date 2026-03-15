"""
v1_4_bear.py - 激进版熊市模块

逻辑：
- BTC > MA200（牛市/震荡）：运行原 v1.4 ���势+MR+动量策略
- BTC < MA200（熊市）：
    1. 资金费率套利（基础层）
    2. 弱势币做空（熊���确认5天后，上限30%）
- 极端负费率（< -0.01%/次）时切回稳定币

验收标准：
- 2022年年化 > 15%
- MaxDD < 15%
- 合并后全期 Sharpe > 2.0
"""
import sys
sys.path.insert(0, r'D:\YZX\alpha-agent\src')
from loguru import logger; logger.remove(); logger.add(sys.stdout, level='WARNING')
import pandas as pd
import numpy as np
from data.fetcher import fetch_all_symbols, fetch_funding_all, build_panel, SYMBOLS_TOP30
from factors.ts_signals_v2 import calc_adx
from factors.volatility import HistoricalVolatility
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics
import time

STABLE_YIELD = 0.05 / 365
SYMS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'AVAXUSDT']
FUNDING_NEG_THRESH = -0.0001   # -0.01%/次
BASE_WEIGHT = 0.2
SHORT_MAX_SINGLE = 0.10        # 单币做空上限10%
SHORT_MAX_TOTAL = 0.30         # 总做空上限30%
BEAR_CONFIRM_DAYS = 5          # 熊市确认天数

# ── 加载���据 ─────────────────────────────────────────────
t0 = time.time()
data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMBOLS_TOP30, use_cache=True)
panel = build_panel(data, funding_data=funding, min_history_days=60)
prices = panel['close'].unstack(level='symbol')
dates_list = sorted(prices.index)
hvol10 = HistoricalVolatility(window=10)
factor = hvol10.compute(panel)

# BTC 技术指标
btc = data['BTCUSDT']
btc_close = btc['close']
btc_ma200 = btc_close.rolling(200).mean()
btc_above_ma200 = (btc_close > btc_ma200).shift(1).fillna(False)
btc_adx20 = calc_adx(btc, window=20).shift(1).fillna(0)
btc_adx14 = calc_adx(btc, window=14).shift(1).fillna(0)

# 牛市确认（v1.4原逻辑）
def make_bull_market(confirm_days=10):
    above_nd = btc_above_ma200.rolling(confirm_days).min().fillna(0).astype(bool)
    adx_ok = (btc_adx20 > 25)
    return (above_nd & adx_ok).shift(1).fillna(False)

bull_market = make_bull_market(10)

# 熊市确认（BTC < MA200 ���续5天）
bear_confirmed = (~btc_above_ma200).rolling(BEAR_CONFIRM_DAYS).min().fillna(0).astype(bool).shift(1).fillna(False)

# ── 资金费率 ──────────────────────────────────���───────────
def get_funding_series(sym):
    try:
        return panel.xs(sym, level='symbol')['funding_rate'].shift(1).fillna(0)
    except Exception:
        return pd.Series(dtype=float)

funding_series = {sym: get_funding_series(sym) for sym in SYMS}

def get_portfolio_funding(date):
    total = 0.0
    count = 0
    for sym in SYMS:
        fr = funding_series[sym]
        if date in fr.index:
            val = fr.loc[date]
            if val < FUNDING_NEG_THRESH:
                return STABLE_YIELD
            total += val
            count += 1
    if count == 0:
        return STABLE_YIELD
    return max((total / count) * 3, STABLE_YIELD)

# ── 做空信号（熊市弱势币）─────���──────────────────────────
def make_short_signal(sym):
    """MACD死��� + ADX>25 + 价格<MA50，ATR×1.5止损"""
    ohlcv = data[sym]
    c = ohlcv['close']
    ef = c.ewm(span=12, adjust=False).mean()
    es = c.ewm(span=26, adjust=False).mean()
    hist = (ef - es) - (ef - es).ewm(span=9, adjust=False).mean()
    ma50 = c.rolling(50).mean()
    adx = calc_adx(ohlcv, window=14)
    # 做空信号：MACD死叉 + ADX>25 + 价格<MA50
    short_sig = ((hist < 0) & (c < ma50) & (adx > 25)).astype(int)
    # 只在熊���确认后激活
    short_sig = short_sig & (~btc_above_ma200).reindex(short_sig.index).fillna(False).astype(int)
    return short_sig.shift(1).fillna(0).astype(int)

short_signals = {sym: make_short_signal(sym) for sym in SYMS}

# ── 牛���趋势信号（v1.4原逻辑）───────────────────────────
def make_trend_sig(sym):
    ohlcv = data[sym]
    c = ohlcv['close']
    ef = c.ewm(span=12, adjust=False).mean()
    es = c.ewm(span=26, adjust=False).mean()
    hist = (ef - es) - (ef - es).ewm(span=9, adjust=False).mean()
    ma200 = c.rolling(200).mean()
    adx = calc_adx(ohlcv, window=20)
    bull_r = bull_market.reindex(adx.index).fillna(False)
    adx_thresh = pd.Series(35.0, index=adx.index)
    adx_thresh[bull_r] = 20.0
    sig = ((hist > 0) & (c > ma200) & (adx > adx_thresh)).astype(int)
    sig = sig & btc_above_ma200.reindex(sig.index).fillna(False)
    return sig.shift(1).fillna(0).astype(int)

def make_mr_sig(sym):
    c = data[sym]['close']
    mid = c.rolling(10).mean()
    lower = mid - 1.5 * c.rolling(10).std()
    pos = pd.Series(0, index=c.index)
    in_t = False
    for i in range(len(pos)):
        if not in_t and c.iloc[i] < lower.iloc[i]:
            in_t = True
        elif in_t and c.iloc[i] > mid.iloc[i]:
            in_t = False
        pos.iloc[i] = 1 if in_t else 0
    pos = (pos.astype(bool) & btc_above_ma200.reindex(pos.index).fillna(False)).astype(int)
    return pos.shift(1).fillna(0).astype(int)

# ── 单币回测 ──────────────���───────────────────────────────
def run_sym(sym, start=None, end=None):
    trend_sig = make_trend_sig(sym)
    mr_sig = make_mr_sig(sym)
    engine_t = TimeSeriesEngine(atr_mult=1.0, use_atr_stop=True)
    engine_m = TimeSeriesEngine(atr_mult=1.0, use_atr_stop=True)
    engine_s = TimeSeriesEngine(atr_mult=1.5, use_atr_stop=True)  # 做空ATR1.5
    r_trend = engine_t.run(trend_sig, data[sym], start_date=start, end_date=end)
    r_mr = engine_m.run(mr_sig, data[sym], start_date=start, end_date=end)
    r_short = engine_s.run(short_signals[sym], data[sym], start_date=start, end_date=end)
    if not isinstance(r_trend, pd.DataFrame) or r_trend.empty:
        return pd.Series(dtype=float)

    rets = []
    cum_ret = 0.0
    for date, row in r_trend.iterrows():
        btc_ok = btc_above_ma200.loc[date] if date in btc_above_ma200.index else False
        is_bull = bull_market.loc[date] if date in bull_market.index else False
        is_bear_confirmed = bear_confirmed.loc[date] if date in bear_confirmed.index else False
        adx_val = btc_adx20.loc[date] if date in btc_adx20.index else 25

        if not btc_ok:
            # ── 熊市逻辑 ──
            # 基础：资金费率套利
            daily_ret = get_portfolio_funding(date) * BASE_WEIGHT

            # 进阶：熊市确认后���空弱势币（上限30%总仓位）
            if is_bear_confirmed and date in r_short.index and r_short.loc[date, 'position'] == 1:
                # 做空收益取反（做空时价格下跌=正收益）
                short_ret = -r_short.loc[date, 'return']  # 反向
                short_weight = min(SHORT_MAX_SINGLE, SHORT_MAX_TOTAL / len(SYMS))
                daily_ret += short_ret * short_weight

            rets.append(daily_ret)
            continue

        # ── 牛市/震���：v1.4原逻辑 ──
        w = BASE_WEIGHT * (2.0 if is_bull else 1.0)
        if row['position'] == 1:
            cum_ret += row['return']
            if cum_ret > 0.05:
                w = min(w * 1.5, BASE_WEIGHT * 2.5)
            rets.append(row['return'] * w)
        elif adx_val < 25 and date in r_mr.index and r_mr.loc[date, 'position'] == 1:
            rets.append(r_mr.loc[date, 'return'] * BASE_WEIGHT * 0.5)
        else:
            cum_ret = 0.0
            dr = STABLE_YIELD
            try:
                df = factor.xs(date, level='timestamp').dropna()
                if len(df) >= 10 and date in prices.index:
                    idx = dates_list.index(date)
                    if idx + 1 < len(dates_list):
                        nd = dates_list[idx + 1]
                        top = df.nlargest(5).index
                        cr = sum((prices.loc[nd, s] - prices.loc[date, s]) / prices.loc[date, s]
                                 for s in top if s in prices.columns) / len(top)
                        dr = 0.5 * cr + 0.5 * STABLE_YIELD
            except Exception:
                pass
            rets.append(dr * BASE_WEIGHT)

    return pd.Series(rets, index=r_trend.index)


def run_portfolio(start=None, end=None):
    all_rets = [run_sym(s, start, end) for s in SYMS]
    return pd.concat([r for r in all_rets if not r.empty], axis=1).fillna(0).sum(axis=1)


def report(ret, label):
    m = full_metrics(ret)
    oos = ret[ret.index >= '2022-01-01']
    mo = full_metrics(oos)
    print(f'\n[{label}]')
    print(f'  全期: Sh={m["sharpe"]:.3f} DD={m["max_drawdown"]:.1%} Ann={m["annual_return"]:.1%}')
    print(f'  OOS:  Sh={mo["sharpe"]:.3f} DD={mo["max_drawdown"]:.1%} Ann={mo["annual_return"]:.1%}')
    for yr in range(2020, 2026):
        r = ret[ret.index.year == yr]
        if len(r) > 5:
            mm = full_metrics(r)
            flag = ''
            if yr == 2022:
                ann_ok = mm['annual_return'] >= 0.15
                dd_ok = mm['max_drawdown'] > -0.15
                flag = ' ✓' if ann_ok and dd_ok else ' ✗'
            print(f'    {yr}: Ann={mm["annual_return"]:+.1%} Sh={mm["sharpe"]:+.3f} DD={mm["max_drawdown"]:.1%}{flag}')
    return m, mo


if __name__ == '__main__':
    print('Running v1.4 + Bear Module...')
    ret = run_portfolio(start='2020-01-01')

    print('\n=== v1.4 + 熊市模块（资金费率 + 做空补位）===')
    m, mo = report(ret, 'v1.4_bear')

    ret_2022 = ret[ret.index.year == 2022]
    m22 = full_metrics(ret_2022)
    print(f'\n2022专项:')
    print(f'  年化={m22["annual_return"]:+.1%} (目标>15%) {"✓" if m22["annual_return"] >= 0.15 else "���"}')
    print(f'  MaxDD={m22["max_drawdown"]:.1%} (目标>-15%) {"✓" if m22["max_drawdown"] > -0.15 else "✗"}')
    print(f'  全期Sharpe={m["sharpe"]:.3f} (目标>2.0) {"✓" if m["sharpe"] >= 2.0 else "✗"}')

    print(f'\nTotal: {time.time()-t0:.1f}s')
