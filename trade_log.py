"""
trade_log.py - 回测交易记录脚本
记录每次开仓/平仓的详细信息
适用于 v1.4 激进版���可修改参数适配稳健版）
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
from pathlib import Path
import time

STABLE_YIELD   = 0.05/365
LENDING_YIELD  = 0.08/365
FUNDING_THRESH = 0.00005
SYMS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','AVAXUSDT']
INITIAL_CAPITAL = 100_000

t0 = time.time()
data    = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
funding = fetch_funding_all(symbols=SYMS, use_cache=True)
panel   = build_panel(data, funding_data=funding, min_history_days=60)

btc             = data['BTCUSDT']
btc_close       = btc['close']
btc_ma200       = btc_close.rolling(200).mean()
btc_above_ma200 = (btc_close > btc_ma200).shift(1).fillna(False)
btc_adx_global  = calc_adx(btc, window=20).shift(1).fillna(0)
above_10d       = btc_above_ma200.rolling(10).min().fillna(0).astype(bool)
bull_market     = (above_10d & (btc_adx_global > 25)).shift(1).fillna(False)

def get_funding_series(sym):
    try:
        return panel.xs(sym, level='symbol')['funding_rate'].shift(1).fillna(0)
    except Exception:
        return pd.Series(dtype=float)

funding_series = {sym: get_funding_series(sym) for sym in SYMS}

def get_bear_return(date):
    daily_rets = []
    for sym in SYMS:
        fr   = funding_series[sym]
        rate = fr.loc[date] if date in fr.index else 0.0
        daily_rets.append(abs(rate)*3 if abs(rate) >= FUNDING_THRESH else LENDING_YIELD)
    return float(np.mean(daily_rets)) if daily_rets else LENDING_YIELD

def make_trend_sig(ohlcv):
    c  = ohlcv['close']
    ef = c.ewm(span=12, adjust=False).mean()
    es = c.ewm(span=26, adjust=False).mean()
    hist = (ef-es) - (ef-es).ewm(span=9, adjust=False).mean()
    ma200 = c.rolling(200).mean()
    adx   = calc_adx(ohlcv, window=20)
    bull_r   = bull_market.reindex(adx.index).fillna(False)
    adx_thr  = pd.Series(35.0, index=adx.index)
    adx_thr[bull_r] = 20.0
    sig = ((hist > 0) & (c > ma200) & (adx > adx_thr)).astype(int)
    sig = sig & btc_above_ma200.reindex(sig.index).fillna(False)
    return sig.shift(1).fillna(0).astype(int)

def make_mr_sig(ohlcv):
    c   = ohlcv['close']
    mid = c.rolling(10).mean()
    lower = mid - 1.5 * c.rolling(10).std()
    pos = pd.Series(0, index=c.index)
    in_t = False
    for i in range(len(pos)):
        if not in_t and c.iloc[i] < lower.iloc[i]: in_t = True
        elif in_t and c.iloc[i] > mid.iloc[i]: in_t = False
        pos.iloc[i] = 1 if in_t else 0
    pos = (pos.astype(bool) & btc_above_ma200.reindex(pos.index).fillna(False)).astype(int)
    return pos.shift(1).fillna(0).astype(int)

# ── 交易记录生成 ──────────────────────────────────────────
all_trades = []   # 每笔交易记录
all_daily  = []   # 每日持仓记录

def run_sym_with_log(sym, base_w=0.2, start='2020-01-01'):
    ohlcv     = data[sym]
    prices    = ohlcv['close']
    trend_sig = make_trend_sig(ohlcv)
    mr_sig    = make_mr_sig(ohlcv)
    engine_t  = TimeSeriesEngine(atr_mult=1.0, use_atr_stop=True)
    engine_m  = TimeSeriesEngine(atr_mult=1.0, use_atr_stop=True)
    r_trend   = engine_t.run(trend_sig, ohlcv, start_date=start)
    r_mr      = engine_m.run(mr_sig,    ohlcv, start_date=start)
    if not isinstance(r_trend, pd.DataFrame) or r_trend.empty:
        return pd.Series(dtype=float)

    rets = []
    cum_ret    = 0.0
    in_trade   = False
    entry_date = None
    entry_price= None
    trade_type = None

    for date, row in r_trend.iterrows():
        adx_val = btc_adx_global.loc[date] if date in btc_adx_global.index else 25
        is_bull = bull_market.loc[date]     if date in bull_market.index     else False
        btc_ok  = btc_above_ma200.loc[date] if date in btc_above_ma200.index else False
        price   = float(prices.loc[date])   if date in prices.index          else 0.0
        w = base_w * (2.0 if is_bull else 1.0)

        # ── 仓位判断 ──
        if not btc_ok:
            position_type = 'BEAR_ARB'
            daily_ret = get_bear_return(date) * base_w
        elif row['position'] == 1:
            position_type = 'TREND'
            cum_ret += row['return']
            if cum_ret > 0.05:
                w = min(w * 1.5, base_w * 2.5)
            daily_ret = row['return'] * w
        elif adx_val < 25 and date in r_mr.index and r_mr.loc[date, 'position'] == 1:
            position_type = 'MR_BB'
            daily_ret = r_mr.loc[date, 'return'] * base_w * 0.5
            cum_ret = 0.0
        else:
            position_type = 'STABLE'
            daily_ret = STABLE_YIELD * base_w
            cum_ret = 0.0

        rets.append(daily_ret)

        # ── 开仓记��� ──
        if position_type == 'TREND' and not in_trade:
            in_trade   = True
            entry_date = date
            entry_price= price
            trade_type = 'TREND_BULL' if is_bull else 'TREND'

        # ── 平仓记录 ���─
        elif in_trade and (position_type != 'TREND' or row.get('stopped', False)):
            exit_date  = date
            exit_price = price
            hold_days  = (exit_date - entry_date).days
            pnl_pct    = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
            stop_flag  = bool(row.get('stopped', False))
            all_trades.append({
                'symbol':      sym,
                'entry_date':  entry_date.date(),
                'exit_date':   exit_date.date(),
                'entry_price': round(entry_price, 4),
                'exit_price':  round(exit_price, 4),
                'hold_days':   hold_days,
                'pnl_pct':     round(pnl_pct * 100, 2),
                'trade_type':  trade_type,
                'stop_loss':   stop_flag,
                'year':        entry_date.year,
            })
            in_trade   = False
            entry_date = None
            cum_ret    = 0.0

        # ── 日记录 ─���
        all_daily.append({
            'date':          date.date(),
            'symbol':        sym,
            'position_type': position_type,
            'daily_ret_pct': round(daily_ret * 100, 4),
            'price':         round(price, 4),
            'is_bull':       is_bull,
            'btc_above_ma':  btc_ok,
        })

    return pd.Series(rets, index=r_trend.index)

# ── 运行 ───────────────────────────────���─────────────────
print('Running trade log backtest...')
all_rets = [run_sym_with_log(s, 0.2, '2020-01-01') for s in SYMS]
portfolio = pd.concat([r for r in all_rets if not r.empty], axis=1).fillna(0).sum(axis=1)

# ── 组合净值（本金10万）──────────────────────────────���────
nav = (1 + portfolio).cumprod() * INITIAL_CAPITAL

# ── 输出汇总 ────────────���────────────────────────────────
trades_df = pd.DataFrame(all_trades)
daily_df  = pd.DataFrame(all_daily)

print(f'\n总交易笔数: {len(trades_df)}')
if not trades_df.empty:
    wins  = (trades_df['pnl_pct'] > 0).sum()
    total = len(trades_df)
    print(f'盈利笔数: {wins} / {total} ({wins/total*100:.1f}%)')
    print(f'平均持仓天数: {trades_df["hold_days"].mean():.1f}天')
    print(f'平均盈亏: {trades_df["pnl_pct"].mean():+.2f}%')
    print(f'最大单笔盈利: {trades_df["pnl_pct"].max():+.2f}%')
    print(f'最大单笔亏损: {trades_df["pnl_pct"].min():+.2f}%')
    stop_n = trades_df['stop_loss'].sum()
    print(f'触发止损次数: {stop_n}')
    print(f'\n当前净值: ${nav.iloc[-1]:,.0f} (初始$100,000)')
    m = full_metrics(portfolio)
    print(f'全期 Sharpe: {m["sharpe"]:.3f}')
    print(f'最大回撤: {m["max_drawdown"]:.1%}')
    print(f'年化收益: {m["annual_return"]:.1%}')

    # 逐年交易统计
    print('\n逐年交易统计:')
    for yr in range(2020, 2027):
        yr_t = trades_df[trades_df['year'] == yr]
        if len(yr_t) == 0: continue
        yr_wins = (yr_t['pnl_pct'] > 0).sum()
        print(f'  {yr}: {len(yr_t)}笔  胜率={yr_wins/len(yr_t)*100:.0f}%  '
              f'均盈亏={yr_t["pnl_pct"].mean():+.1f}%  '
              f'止损={yr_t["stop_loss"].sum()}次')

# ── 保存文件 ────────────���────────────────────────────────
out_dir = Path(r'D:\YZX\alpha-agent-aggressive\output')
out_dir.mkdir(parents=True, exist_ok=True)

trades_path = out_dir / 'trade_log.csv'
daily_path  = out_dir / 'daily_log.csv'
nav_path    = out_dir / 'nav_log.csv'

trades_df.to_csv(trades_path, index=False, encoding='utf-8-sig')
daily_df.to_csv(daily_path,   index=False, encoding='utf-8-sig')
nav.to_csv(nav_path)

print(f'\n[Saved] {trades_path}')
print(f'[Saved] {daily_path}')
print(f'[Saved] {nav_path}')
print(f'\nTotal: {time.time()-t0:.1f}s')
