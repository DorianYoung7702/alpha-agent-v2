"""
Factor Base Class - 因子基类
所有因子必须继承此类，强制 shift(1) 防未来函数
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from loguru import logger


class BaseFactor(ABC):
    """
    因子基类
    子类实现 _compute() 方法
    基类自动处理：shift(1)、截面标准化、异常值处理
    """
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.params = {}

    @abstractmethod
    def _compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        计算原始因子值
        panel: MultiIndex (timestamp, symbol) DataFrame
        返回: Series with same index
        注意：此处可以用 t 日数据，shift 在基类自动处理
        """
        pass

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        对外接口：计算 + shift(1) + 截面标准化
        """
        try:
            raw = self._compute(panel)
            # 强制 shift(1)：信号基于昨日数据，今日执行
            shifted = raw.groupby(level="symbol").shift(1)
            # 截面标准化（���日对所有币种做 z-score）
            normalized = self._cross_sectional_zscore(shifted)
            return normalized.rename(self.name)
        except Exception as e:
            logger.error(f"Factor {self.name} compute error: {e}")
            raise

    def _cross_sectional_zscore(self, factor: pd.Series) -> pd.Series:
        """
        截面 z-score 标准化
        每个时间截面（日）对所有币种归一化
        """
        def zscore(x):
            std = x.std()
            if std == 0 or pd.isna(std):
                return x * 0
            return (x - x.mean()) / std

        return factor.groupby(level="timestamp").transform(zscore)

    def _winsorize(self, series: pd.Series, n_std: float = 3.0) -> pd.Series:
        """
        3sigma 去极值
        """
        mean = series.mean()
        std = series.std()
        return series.clip(mean - n_std * std, mean + n_std * std)

    def compute_ic(
        self,
        factor: pd.Series,
        forward_returns: pd.Series,
        method: str = "spearman"
    ) -> pd.Series:
        """
        计算每日 IC（因子与未来收益的相关性）
        """
        combined = pd.concat([factor.rename("factor"), forward_returns.rename("ret")], axis=1).dropna()
        def daily_ic(g):
            if len(g) < 3:
                return np.nan
            if method == "spearman":
                return g["factor"].corr(g["ret"], method="spearman")
            return g["factor"].corr(g["ret"])
        return combined.groupby(level="timestamp").apply(daily_ic)

    def __repr__(self):
        return f"Factor({self.name}): {self.description}"
