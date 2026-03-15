import sys
sys.path.insert(0, r'D:\YZX\alpha-agent\src')
from loguru import logger; logger.remove(); logger.add(sys.stdout, level='WARNING')
import pandas as pd
import numpy as np
from data.fetcher import fetch_all_symbols, fetch_funding_all, build_panel, SYMBOLS_TOP30
from factors.ts_signals_v2 import calc_adx
from backtest.engine_ts import TimeSeriesEngine
from backtest.metrics import full_metrics

STABLE_YIELD=0.05/365; LENDING_YIELD=0.08/365; FUNDING_THRESH=0.00005
SYMS=['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','AVAXUSDT']
MAX_W=0.30; MAX_TW=1.50

data=fetch_all_symbols(symbols=SYMBOLS_TOP30,use_cache=True)
funding=fetch_funding_all(symbols=SYMS,use_cache=True)
panel=build_panel(data,funding_data=funding,min_history_days=60)
btc=data['BTCUSDT']; btc_close=btc['close']
btc_ma200=btc_close.rolling(200).mean()
btc_above=(btc_close>btc_ma200).shift(1).fillna(False)
btc_adx=calc_adx(btc,window=20).shift(1).fillna(0)
above_10d=btc_above.rolling(10).min().fillna(0).astype(bool)
bull=(above_10d&(btc_adx>25)).shift(1).fillna(False)

def fs(sym):
    try: return panel.xs(sym,level='symbol')['funding_rate'].shift(1).fillna(0)
    except: return pd.Series(dtype=float)
fsr={s:fs(s) for s in SYMS}

def bear_ret(date):
    r=[abs(fsr[s].loc[date])*3 if date in fsr[s].index and abs(fsr[s].loc[date])>=FUNDING_THRESH else LENDING_YIELD for s in SYMS]
    return float(np.mean(r))

def run_combo(adx_base, atr):
    all_rets=[]
    for sym in SYMS:
        ohlcv=data[sym]; c=ohlcv['close']
        ef=c.ewm(span=20,adjust=False).mean(); es=c.ewm(span=60,adjust=False).mean()
        ma200=c.rolling(200).mean(); adx=calc_adx(ohlcv,window=14)
        bull_r=bull.reindex(adx.index).fillna(False)
        adx_thr=pd.Series(float(adx_base),index=adx.index)
        adx_thr[bull_r]=max(adx_base-10,10)
        sig=((ef>es)&(c>ma200)&(adx>adx_thr)).astype(int)&btc_above.reindex(adx.index).fillna(False)
        sig=sig.shift(1).fillna(0).astype(int)
        engine=TimeSeriesEngine(atr_mult=atr,use_atr_stop=True)
        r=engine.run(sig,ohlcv,start_date='2020-01-01')
        if not isinstance(r,pd.DataFrame) or r.empty: continue
        rets=[]; cum=0.0
        for date,row in r.iterrows():
            ib=bull.loc[date] if date in bull.index else False
            bok=btc_above.loc[date] if date in btc_above.index else False
            w=min(0.2*(2.0 if ib else 1.0),MAX_W)
            if not bok: rets.append(bear_ret(date)*0.2); cum=0.0
            elif row['position']==1:
                cum+=row['return']
                if cum>0.05: w=min(w*1.5,MAX_W)
                rets.append(row['return']*w)
            else: rets.append(STABLE_YIELD*0.2); cum=0.0
        all_rets.append(pd.Series(rets,index=r.index))
    if not all_rets: return pd.Series(dtype=float)
    combined=pd.concat(all_rets,axis=1).fillna(0)
    scale=combined.abs().sum(axis=1).apply(lambda x: min(1.0,MAX_TW/x) if x>MAX_TW else 1.0)
    return combined.multiply(scale,axis=0).sum(axis=1)

targets={2020:0.60,2021:1.50,2022:0.0,2023:0.80,2024:0.60,2025:0.15}
print('ADX    ATR    Sh      MaxDD    OOSSh   达标  2020')
print('-'*55)
for adx in [28,30,32,35]:
    for atr in [0.7,0.8,1.0]:
        ret=run_combo(adx,atr)
        if ret.empty: continue
        m=full_metrics(ret)
        oos=ret[ret.index>='2022-01-01']
        mo=full_metrics(oos)
        passed=0
        yr2020_ann=0
        for yr in range(2020,2026):
            r_yr=ret[ret.index.year==yr]
            if len(r_yr)>5:
                ann=full_metrics(r_yr)['annual_return']
                if ann>=targets.get(yr,0): passed+=1
                if yr==2020: yr2020_ann=ann
        print('{:^6} {:^5} {:^7.3f} {:^8.1%} {:^7.3f} {:^4}/6 {:+.1%}'.format(
            adx, atr, m['sharpe'], m['max_drawdown'], mo['sharpe'], passed, yr2020_ann))
