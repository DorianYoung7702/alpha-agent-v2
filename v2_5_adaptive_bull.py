"""
v2_5_adaptive_bull.py - v2.5 自适应ADX + 更激进牛市阈值
在 v2.4 基础上：牛市时ADX阈值���幅从-10改为-15（更激进）
目标：年化100%+，同时保持OOSSh>1.75, MaxDD<-33%
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

STABLE_YIELD=0.05/365; LENDING_YIELD=0.08/365; FUNDING_THRESH=0.00005
SYMS=['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','AVAXUSDT']
EMA_FAST=20; EMA_SLOW=60; ATR_MULT=0.7
MAX_SINGLE_WEIGHT=0.30; MAX_TOTAL_WEIGHT=1.50
BULL_ADX_DROP=15  # 牛市ADX降幅（v2.4是10）

t0=time.time()
data=fetch_all_symbols(symbols=SYMBOLS_TOP30,use_cache=True)
funding=fetch_funding_all(symbols=SYMS,use_cache=True)
panel=build_panel(data,funding_data=funding,min_history_days=60)

btc=data['BTCUSDT']; btc_close=btc['close']
btc_ma200=btc_close.rolling(200).mean()
btc_above_ma200=(btc_close>btc_ma200).shift(1).fillna(False)
btc_adx_global=calc_adx(btc,window=20).shift(1).fillna(0)
above_10d=btc_above_ma200.rolling(10).min().fillna(0).astype(bool)
bull_market=(above_10d&(btc_adx_global>25)).shift(1).fillna(False)

def get_fs(sym):
    try: return panel.xs(sym,level='symbol')['funding_rate'].shift(1).fillna(0)
    except: return pd.Series(dtype=float)
fsr={s:get_fs(s) for s in SYMS}

def bear_ret(date):
    r=[abs(fsr[s].loc[date])*3 if date in fsr[s].index and abs(fsr[s].loc[date])>=FUNDING_THRESH else LENDING_YIELD for s in SYMS]
    return float(np.mean(r))

def make_sig(sym):
    ohlcv=data[sym]; c=ohlcv['close']
    ef=c.ewm(span=EMA_FAST,adjust=False).mean(); es=c.ewm(span=EMA_SLOW,adjust=False).mean()
    ma200=c.rolling(200).mean(); adx=calc_adx(ohlcv,window=14)
    adx_med60=(adx.rolling(60).median()*1.2).clip(20,50)
    bull_r=bull_market.reindex(adx.index).fillna(False)
    adx_thr=adx_med60.copy()
    adx_thr[bull_r]=(adx_med60[bull_r]-BULL_ADX_DROP).clip(10,50)
    sig=((ef>es)&(c>ma200)&(adx>adx_thr)).astype(int)
    sig=sig&btc_above_ma200.reindex(sig.index).fillna(False)
    return sig.shift(1).fillna(0).astype(int)

def run_sym(sym,base_w=0.2,start=None):
    sig=make_sig(sym); engine=TimeSeriesEngine(atr_mult=ATR_MULT,use_atr_stop=True)
    r=engine.run(sig,data[sym],start_date=start)
    if not isinstance(r,pd.DataFrame) or r.empty: return pd.Series(dtype=float)
    rets=[]; cum=0.0
    for date,row in r.iterrows():
        ib=bull_market.loc[date] if date in bull_market.index else False
        bok=btc_above_ma200.loc[date] if date in btc_above_ma200.index else False
        w=min(base_w*(2.0 if ib else 1.0),MAX_SINGLE_WEIGHT)
        if not bok: rets.append(bear_ret(date)*base_w); cum=0.0
        elif row['position']==1:
            cum+=row['return']
            if cum>0.05: w=min(w*1.5,MAX_SINGLE_WEIGHT)
            rets.append(row['return']*w)
        else: rets.append(STABLE_YIELD*base_w); cum=0.0
    return pd.Series(rets,index=r.index)

def run_portfolio(start='2020-01-01'):
    all_rets=[run_sym(s,0.2,start) for s in SYMS]
    combined=pd.concat([r for r in all_rets if not r.empty],axis=1).fillna(0)
    scale=combined.abs().sum(axis=1).apply(lambda x: min(1.0,MAX_TOTAL_WEIGHT/x) if x>MAX_TOTAL_WEIGHT else 1.0)
    return combined.multiply(scale,axis=0).sum(axis=1)

targets={2020:0.60,2021:1.50,2022:0.0,2023:0.80,2024:0.60,2025:0.15}
ret=run_portfolio('2020-01-01')
m=full_metrics(ret)
oos=ret[ret.index>='2022-01-01']; mo=full_metrics(oos)
passed=0
print('\n[v2.5 自适应ADX + 牛���降幅%d]'%BULL_ADX_DROP)
print('  全期: Sh=%.3f DD=%.1f%% Ann=%.1f%%'%(m['sharpe'],m['max_drawdown']*100,m['annual_return']*100))
print('  OOS:  Sh=%.3f DD=%.1f%% Ann=%.1f%%'%(mo['sharpe'],mo['max_drawdown']*100,mo['annual_return']*100))
for yr in range(2020,2026):
    r=ret[ret.index.year==yr]
    if len(r)>5:
        mm=full_metrics(r); t=targets.get(yr,0)
        f='✓' if mm['annual_return']>=t else '✗'
        if mm['annual_return']>=t: passed+=1
        print('    %d: Ann=%+.1f%% Sh=%+.3f DD=%.1f%% %s'%(yr,mm['annual_return']*100,mm['sharpe'],mm['max_drawdown']*100,f))
print('  达标: %d/6'%passed)
print('  验收: OOSSh=%.3f %s MaxDD=%.1f%% %s 达标=%d/6 %s'%(
    mo['sharpe'],'✅' if mo['sharpe']>=1.75 else '❌',
    m['max_drawdown']*100,'✅' if m['max_drawdown']>-0.33 else '❌',
    passed,'✅' if passed==6 else '❌'))
lines=['# v2.5 自适应ADX 牛市降幅%d'%BULL_ADX_DROP,
    '\n**时间**: '+pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
    '\n## 核心指标','| 指标 | 全期 | OOS |','|------|------|-----|',
    '| Sharpe | %.3f | %.3f |'%(m['sharpe'],mo['sharpe']),
    '| MaxDD | %.1f%% | %.1f%% |'%(m['max_drawdown']*100,mo['max_drawdown']*100),
    '| 年化 | %.1f%% | %.1f%% |'%(m['annual_return']*100,mo['annual_return']*100),
    '\n## 验收',
    '- OOSSh=%.3f %s'%(mo['sharpe'],'✅' if mo['sharpe']>=1.75 else '❌'),
    '- MaxDD=%.1f%% %s'%(m['max_drawdown']*100,'✅' if m['max_drawdown']>-0.33 else '❌'),
    '- 达标=%d/6 %s'%(passed,'✅' if passed==6 else '❌'),
]
Path(r'D:\YZX\shared\results\aggressive_latest.md').write_text('\n'.join(lines),encoding='utf-8')
print('  [Saved]'); print('Total: %.1fs'%(time.time()-t0))
