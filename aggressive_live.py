"""
aggressive_live.py - v1.4 激进版 OKX 实盘接入

模式：
- DRY_RUN=True：干跑，只打印信号不下单
- DRY_RUN=False：真实下单

运行：python aggressive_live.py
定时：每日 UTC 00:05 自动执行（或手动触发）
"""
import sys
sys.path.insert(0, r'D:\YZX\alpha-agent\src')
sys.path.insert(0, r'D:\YZX\alpha-agent-aggressive\src')
from loguru import logger
logger.remove()
logger.add(sys.stdout, level='INFO',
           format='<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}')
logger.add(r'D:\YZX\alpha-agent-aggressive\logs\live_{time:YYYY-MM-DD}.log',
           level='DEBUG', rotation='1 day')

import pandas as pd
import numpy as np
import requests
import hmac
import hashlib
import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from data.fetcher import fetch_klines, fetch_all_symbols, fetch_funding_all, build_panel, SYMBOLS_TOP30
from factors.ts_signals_v2 import calc_adx
from backtest.engine_ts import TimeSeriesEngine

# ─��� 配置 ───────────────────────────────────────���──────────
DRY_RUN = True   # True=干跑 False=真实下单

API_KEY    = 'd9d9a419-dd5c-42e1-8b26-c8edac3c6acd'
SECRET_KEY = 'CB52E3AB1E665615D185B0F7F402591B'
PASSPHRASE = ''
BASE_URL   = 'https://www.okx.com'

SYMS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'BNB-USDT', 'AVAX-USDT']
SYMS_BINANCE = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'AVAXUSDT']
BASE_WEIGHT  = 0.2   # 基础仓位 20%
BULL_MULT    = 2.0   # 牛市倍数
MOM_THRESH   = 0.05  # 动量加仓门槛
MOM_ADD      = 0.5   # 动量加仓量

Path(r'D:\YZX\alpha-agent-aggressive\logs').mkdir(parents=True, exist_ok=True)

# ── OKX API ──────────────────────────────────────���────────
def _sign(timestamp, method, path, body=''):
    msg = f'{timestamp}{method}{path}{body}'
    mac = hmac.new(SECRET_KEY.encode(), msg.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

def _headers(method, path, body=''):
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    return {
        'OK-ACCESS-KEY':        API_KEY,
        'OK-ACCESS-SIGN':       _sign(ts, method, path, body),
        'OK-ACCESS-TIMESTAMP':  ts,
        'OK-ACCESS-PASSPHRASE': PASSPHRASE,
        'Content-Type':         'application/json',
    }

def get_account_balance():
    path = '/api/v5/account/balance'
    r = requests.get(BASE_URL + path, headers=_headers('GET', path), timeout=10)
    data = r.json()
    if data.get('code') != '0':
        logger.error(f'Balance error: {data}')
        return {}
    balances = {}
    for detail in data['data'][0]['details']:
        if float(detail['availBal']) > 0:
            balances[detail['ccy']] = float(detail['availBal'])
    return balances

def get_ticker(inst_id):
    path = f'/api/v5/market/ticker?instId={inst_id}'
    r = requests.get(BASE_URL + path, timeout=10)
    data = r.json()
    if data.get('code') != '0' or not data['data']:
        return None
    return float(data['data'][0]['last'])

def place_order(inst_id, side, usdt_amount, dry_run=True):
    """
    side: 'buy' or 'sell'
    usdt_amount: USDT 金额
    """
    price = get_ticker(inst_id)
    if price is None:
        logger.error(f'Cannot get price for {inst_id}')
        return False

    # 计算数量
    ccy = inst_id.split('-')[0]
    qty = usdt_amount / price

    if dry_run:
        logger.info(f'[DRY] {side.upper()} {inst_id}: {qty:.6f} @ {price:.2f} = ${usdt_amount:.2f}')
        return True

    path = '/api/v5/trade/order'
    body = json.dumps({
        'instId':  inst_id,
        'tdMode': 'cash',
        'side':    side,
        'ordType': 'market',
        'sz':      f'{qty:.6f}',
        'tgtCcy':  'base_ccy',
    })
    r = requests.post(BASE_URL + path,
                      headers=_headers('POST', path, body),
                      data=body, timeout=10)
    result = r.json()
    if result.get('code') != '0':
        logger.error(f'Order failed {inst_id}: {result}')
        return False
    logger.success(f'[ORDER] {side.upper()} {inst_id} {qty:.6f} @ {price:.2f}')
    return True

# ── 信号计算（与 aggressive_v4.py 完全相同）────────────────
def compute_signals():
    logger.info('Fetching latest data...')
    data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=False)  # 强制更新
    btc = data['BTCUSDT']
    btc_close   = btc['close']
    btc_ma200   = btc_close.rolling(200).mean()
    btc_above   = (btc_close > btc_ma200).shift(1).fillna(False)
    btc_adx_g   = calc_adx(btc, window=20).shift(1).fillna(0)

    # 牛市确认（10天连续 + ADX>25）
    above_10d   = btc_above.rolling(10).min().fillna(0).astype(bool)
    adx_ok      = btc_adx_g > 25
    bull_market = (above_10d & adx_ok).shift(1).fillna(False)

    signals = {}
    today = btc_close.index[-1]

    for sym_b, sym_okx in zip(SYMS_BINANCE, SYMS):
        ohlcv = data[sym_b]
        c  = ohlcv['close']
        ef = c.ewm(span=12, adjust=False).mean()
        es = c.ewm(span=26, adjust=False).mean()
        hist = (ef - es) - (ef - es).ewm(span=9, adjust=False).mean()
        ma200 = c.rolling(200).mean()
        adx   = calc_adx(ohlcv, window=20)

        # 牛市ADX门槛降至20
        bull_r   = bull_market.reindex(adx.index).fillna(False)
        adx_thr  = pd.Series(35.0, index=adx.index)
        adx_thr[bull_r] = 20.0

        sig = ((hist > 0) & (c > ma200) & (adx > adx_thr)).astype(int)
        sig = sig & btc_above.reindex(sig.index).fillna(False)
        sig = sig.shift(1).fillna(0).astype(int)

        today_sig  = int(sig.loc[today]) if today in sig.index else 0
        is_bull    = bool(bull_market.loc[today]) if today in bull_market.index else False
        btc_ok     = bool(btc_above.loc[today])   if today in btc_above.index   else False
        curr_price = float(c.loc[today])

        # 仓位权重计算
        if not btc_ok:
            weight = 0.0  # 熊市空仓（实盘v1.4无熊市补位，仅空仓）
        elif today_sig == 1:
            w = BASE_WEIGHT * (BULL_MULT if is_bull else 1.0)
            weight = min(w, 0.40)  # 单币最大40%
        else:
            weight = 0.0

        signals[sym_okx] = {
            'signal':    today_sig,
            'is_bull':   is_bull,
            'btc_above': btc_ok,
            'weight':    weight,
            'price':     curr_price,
            'date':      str(today.date()),
        }
        logger.info(f'{sym_okx}: sig={today_sig} bull={is_bull} btc_ok={btc_ok} weight={weight:.0%} price={curr_price:.2f}')

    return signals

# ── 仓位管��� ─────────────────────────────────────────���────
def rebalance(signals, dry_run=True):
    logger.info('Fetching account balance...')
    if dry_run:
        total_usdt = 10000.0  # 干跑假设1万USDT
        current_holdings = {sym: 0.0 for sym in SYMS}  # 假设空仓
        logger.info(f'[DRY] Total USDT: ${total_usdt:.2f}')
    else:
        balances = get_account_balance()
        logger.info(f'Balances: {balances}')
        total_usdt = balances.get('USDT', 0.0)
        # 估算当前持仓价值
        current_holdings = {}
        for sym in SYMS:
            ccy = sym.split('-')[0]
            qty = balances.get(ccy, 0.0)
            price = get_ticker(sym) or 0.0
            current_holdings[sym] = qty * price
        total_portfolio = total_usdt + sum(current_holdings.values())
        total_usdt = total_portfolio  # 用总资产计算目标仓位

    logger.info(f'Total portfolio: ${total_usdt:.2f}')
    logger.info('--- 目标仓位 ---')
    for sym, info in signals.items():
        target_usdt = total_usdt * info['weight']
        current_usdt = current_holdings.get(sym, 0.0)
        diff = target_usdt - current_usdt
        action = 'HOLD'
        if diff > total_usdt * 0.01:   # 差异>1%才调仓
            action = 'BUY'
        elif diff < -total_usdt * 0.01:
            action = 'SELL'
        logger.info(f'  {sym}: target={info["weight"]:.0%} (${target_usdt:.0f}) '
                    f'current=${current_usdt:.0f} diff=${diff:+.0f} → {action}')
        if action == 'BUY' and diff > 0:
            place_order(sym, 'buy',  abs(diff), dry_run)
        elif action == 'SELL' and diff < 0:
            place_order(sym, 'sell', abs(diff), dry_run)

    # 记录执行日志
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'dry_run':   dry_run,
        'signals':   signals,
        'total_usdt': total_usdt,
    }
    log_path = Path(r'D:\YZX\alpha-agent-aggressive\logs\live_trades.json')
    history = []
    if log_path.exists():
        try:
            history = json.loads(log_path.read_text())
        except Exception:
            history = []
    history.append(log_entry)
    log_path.write_text(json.dumps(history, indent=2, ensure_ascii=False))
    logger.info(f'[Saved] {log_path}')


if __name__ == '__main__':
    logger.info(f'=== OUROBOROS v1.4 激进版实盘 | DRY_RUN={DRY_RUN} ===')
    logger.info(f'日期: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    signals = compute_signals()
    rebalance(signals, dry_run=DRY_RUN)

    logger.info('=== 执行完成 ===')
