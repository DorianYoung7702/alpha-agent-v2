"""
Metrics - 回测指标计算
Sharpe, MaxDrawdown, IC, ICIR, 年化收益等
"""
import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger


def calc_sharpe(returns: pd.Series, annualize: int = 252) -> float:
    """年化 Sharpe Ratio"""
    if returns.empty or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(annualize))


def calc_max_drawdown(returns: pd.Series) -> float:
    """最大回撤"""
    if returns.empty:
        return 0.0
    cumret = (1 + returns).cumprod()
    rolling_max = cumret.cummax()
    drawdown = (cumret - rolling_max) / rolling_max
    return float(drawdown.min())


def calc_annual_return(returns: pd.Series, annualize: int = 252) -> float:
    """年化收益率"""
    if returns.empty:
        return 0.0
    total = (1 + returns).prod()
    n = len(returns)
    return float(total ** (annualize / n) - 1)


def calc_win_rate(returns: pd.Series) -> float:
    """胜率"""
    if returns.empty:
        return 0.0
    return float((returns > 0).mean())


def calc_profit_loss_ratio(returns: pd.Series) -> float:
    """盈亏比"""
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if losses.empty or wins.empty:
        return 0.0
    return float(wins.mean() / abs(losses.mean()))


def calc_ic_series(
    factor: pd.Series,
    forward_returns: pd.Series,
    method: str = "spearman"
) -> pd.Series:
    """
    逐日计算 IC
    factor: MultiIndex (timestamp, symbol)
    forward_returns: MultiIndex (timestamp, symbol)，t 日持有 1 日的未来收益
    """
    combined = pd.concat(
        [factor.rename("factor"), forward_returns.rename("fwd_ret")],
        axis=1
    ).dropna()

    def daily_ic(g):
        if len(g) < 3:
            return np.nan
        if method == "spearman":
            corr, _ = stats.spearmanr(g["factor"], g["fwd_ret"])
        else:
            corr = g["factor"].corr(g["fwd_ret"])
        return corr

    return combined.groupby(level="timestamp").apply(daily_ic).dropna()


def calc_icir(ic_series: pd.Series) -> float:
    """ICIR = IC均值 / IC标准差"""
    if ic_series.empty or ic_series.std() == 0:
        return 0.0
    return float(ic_series.mean() / ic_series.std())


def calc_forward_returns(panel: pd.DataFrame, periods: int = 1) -> pd.Series:
    """
    计算未来 N 日收益率（用于 IC 计算）
    注意：这是真实的未来收益，只用于事后评估，不用于信号生成
    """
    close = panel["close"].unstack(level="symbol")
    fwd = close.shift(-periods) / close - 1
    return fwd.stack().rename("fwd_ret")


def full_metrics(
    returns: pd.Series,
    factor: pd.Series = None,
    panel: pd.DataFrame = None,
    factor_name: str = "unknown"
) -> dict:
    """
    ���算完整指标集
    """
    sharpe = calc_sharpe(returns)
    max_dd = calc_max_drawdown(returns)
    annual_ret = calc_annual_return(returns)
    win_rate = calc_win_rate(returns)
    pl_ratio = calc_profit_loss_ratio(returns)

    ic_mean, ic_std, icir = 0.0, 0.0, 0.0
    if factor is not None and panel is not None:
        try:
            fwd_ret = calc_forward_returns(panel)
            ic_series = calc_ic_series(factor, fwd_ret)
            ic_mean = float(ic_series.mean())
            ic_std = float(ic_series.std())
            icir = calc_icir(ic_series)
        except Exception as e:
            logger.warning(f"IC calc failed: {e}")

    metrics = {
        "factor_name": factor_name,
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "annual_return": round(annual_ret, 4),
        "win_rate": round(win_rate, 4),
        "pl_ratio": round(pl_ratio, 4),
        "ic_mean": round(ic_mean, 4),
        "ic_std": round(ic_std, 4),
        "icir": round(icir, 4),
        "n_days": len(returns)
    }

    # 判断是否通过验证标准（MaxDD放宽至50%）
    # 时序趋势跟踪验证标准：Sharpe>1.5, MaxDD<40%
    metrics["passed"] = (
        sharpe > 1.5 and
        max_dd > -0.40
    )

    return metrics


def print_metrics(metrics: dict):
    """格式化打印指标"""
    status = "✓ PASSED" if metrics.get("passed") else "✗ FAILED"
    logger.info(f"""\n{'='*50}
因子名称：{metrics['factor_name']}
Sharpe：{metrics['sharpe']}
最大回撤：{metrics['max_drawdown']:.1%}
年化收益：{metrics['annual_return']:.1%}
胜率���{metrics['win_rate']:.1%}
盈亏比：{metrics['pl_ratio']:.2f}
IC均值：{metrics['ic_mean']:.4f}
ICIR：{metrics['icir']:.4f}
交易日数：{metrics['n_days']}
结论：{status}
{'='*50}""")
