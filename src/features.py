"""
特征工程模块
输入: 历史销量 DataFrame
输出: 带特征的 DataFrame（用于 LightGBM 训练）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.country_config import HOLIDAYS, PLATFORM_PROMOS, COUNTRY_PROFILE


def add_lag_features(df: pd.DataFrame, target_col: str = "sales_qty") -> pd.DataFrame:
    """滞后特征 + 滚动统计"""
    df = df.sort_values("date").copy()
    lags = [1, 2, 3, 7, 14, 21, 28]
    for lag in lags:
        df[f"lag_{lag}d"] = df[target_col].shift(lag)

    for window in [7, 14, 28]:
        df[f"roll_mean_{window}d"] = df[target_col].shift(1).rolling(window).mean()
        df[f"roll_std_{window}d"]  = df[target_col].shift(1).rolling(window).std()
        df[f"roll_max_{window}d"]  = df[target_col].shift(1).rolling(window).max()

    # 同期对比（上周同天、上月同天）
    df["lag_7d_yoy_ratio"] = df[target_col].shift(7) / (df[target_col].shift(14) + 1)
    df["lag_28d_trend"]    = (df[f"roll_mean_7d"] - df[f"roll_mean_28d"]) / (df[f"roll_mean_28d"] + 1)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """时间特征"""
    df = df.copy()
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["dayofmonth"] = df["date"].dt.day
    df["dayofyear"]  = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]      = df["date"].dt.month
    df["quarter"]    = df["date"].dt.quarter
    df["year"]       = df["date"].dt.year
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # 循环编码（避免 12→1 的跳跃性）
    df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofweek_sin"]  = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"]  = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


def _days_to_event(dates: pd.DatetimeIndex, event_dates: list[str]) -> np.ndarray:
    """距最近节日的天数（负=节日前，正=节日后）"""
    event_ts = [pd.Timestamp(d) for d in event_dates]
    result = np.full(len(dates), 999, dtype=float)
    for i, d in enumerate(dates):
        diffs = [(d - e).days for e in event_ts]
        # 取绝对值最小的
        min_abs_idx = int(np.argmin([abs(x) for x in diffs]))
        result[i] = diffs[min_abs_idx]
    return result


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """节假日距离特征（用于 LightGBM，Prophet 直接用 holidays 参数）"""
    df = df.copy()

    for country, holidays in HOLIDAYS.items():
        mask = df["country"] == country
        if not mask.any():
            continue
        country_dates = pd.DatetimeIndex(df.loc[mask, "date"])

        for h in holidays:
            feat_name = f"days_to_{h['name']}"
            days = _days_to_event(country_dates, h["dates"])
            df.loc[mask, feat_name] = days
            # 二值：节日窗口内
            df.loc[mask, f"in_{h['name']}_window"] = (
                (days >= -h["pre_days"]) & (days <= h["post_days"])
            ).astype(int)

    # 通用节假日 boost（直接使用生成数据中的字段）
    if "holiday_boost" in df.columns:
        df["holiday_active"] = (df["holiday_boost"] > 1.05).astype(int)
        df["holiday_intensity"] = df["holiday_boost"] - 1.0

    return df


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """大促特征"""
    df = df.copy()
    df["is_promo_day"] = 0
    df["promo_name"] = "none"
    df["days_to_next_promo"] = 999

    for year in df["date"].dt.year.unique():
        for month, day, name, duration, _ in PLATFORM_PROMOS:
            try:
                start = pd.Timestamp(year=int(year), month=month, day=day)
            except ValueError:
                continue
            end = start + pd.Timedelta(days=duration - 1)
            mask = (df["date"] >= start) & (df["date"] <= end)
            df.loc[mask, "is_promo_day"] = 1
            df.loc[mask, "promo_name"] = name

    # 距下一次大促天数（向量化计算，避免逐行循环）
    promo_dates_all = sorted(set(
        pd.Timestamp(year=int(y), month=month, day=day)
        for y in df["date"].dt.year.unique()
        for month, day, _, _, _ in PLATFORM_PROMOS
    ))
    promo_ts = np.array([d.value for d in promo_dates_all])
    date_vals = df["date"].values.astype("int64")

    def _days_to_next(dv: np.ndarray) -> np.ndarray:
        result = np.full(len(dv), 999, dtype=float)
        for i, d in enumerate(dv):
            future = promo_ts[promo_ts >= d]
            if len(future) > 0:
                result[i] = (future[0] - d) / 86_400_000_000_000  # ns → days
        return result

    df["days_to_next_promo"] = _days_to_next(date_vals)

    return df


def add_country_features(df: pd.DataFrame) -> pd.DataFrame:
    """国家级消费特征（编码为数值）"""
    df = df.copy()
    df["price_sensitivity"]  = df["country"].map({c: p["price_sensitivity"]  for c, p in COUNTRY_PROFILE.items()})
    df["promo_sensitivity"]  = df["country"].map({c: p["promo_sensitivity"]  for c, p in COUNTRY_PROFILE.items()})
    df["base_demand_level"]  = df["country"].map({c: p["base_demand_multiplier"] for c, p in COUNTRY_PROFILE.items()})
    df["weekend_boost_rate"] = df["country"].map({c: p["weekend_boost"] for c, p in COUNTRY_PROFILE.items()})
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "sales_qty",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    完整特征矩阵构建入口
    group_cols: 分组滞后计算的维度，默认 ['country', 'sku_id']
    """
    if group_cols is None:
        group_cols = ["country", "sku_id"]

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(group_cols + ["date"])

    # 分组计算滞后特征
    parts = []
    for key, grp in df.groupby(group_cols, sort=False):
        grp = add_lag_features(grp, target_col)
        parts.append(grp)
    df = pd.concat(parts, ignore_index=True)

    df = add_time_features(df)
    df = add_country_features(df)

    # 节假日特征（按国家分组处理，promo全局处理）
    country_parts = []
    for country, grp in df.groupby("country", sort=False):
        grp = add_holiday_features(grp)
        country_parts.append(grp)
    df = pd.concat(country_parts, ignore_index=True)

    # 大促特征（按行计算，较慢但准确）
    df = add_promo_features(df)

    # 品类 label encode
    cat_map = {c: i for i, c in enumerate(sorted(df["category"].unique()))}
    df["category_enc"] = df["category"].map(cat_map)

    country_map = {c: i for i, c in enumerate(sorted(df["country"].unique()))}
    df["country_enc"] = df["country"].map(country_map)

    return df


LGBM_FEATURE_COLS = [
    # 时间
    "dayofweek", "dayofmonth", "month", "quarter", "year",
    "is_weekend", "is_month_end", "weekofyear",
    "month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos",
    # 滞后
    "lag_1d", "lag_2d", "lag_3d", "lag_7d", "lag_14d", "lag_21d", "lag_28d",
    # 滚动
    "roll_mean_7d", "roll_mean_14d", "roll_mean_28d",
    "roll_std_7d",  "roll_std_28d",
    "roll_max_7d",  "roll_max_28d",
    "lag_28d_trend",
    # 节假日
    "holiday_active", "holiday_intensity",
    # 大促
    "is_promo_day", "days_to_next_promo",
    # 国家特征
    "price_sensitivity", "promo_sensitivity", "base_demand_level", "weekend_boost_rate",
    # 编码
    "category_enc", "country_enc",
]
