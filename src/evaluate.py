"""
预测评估模块
"""

import pandas as pd
import numpy as np


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual    = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    mask_nonzero = actual > 0
    mae   = np.mean(np.abs(actual - predicted))
    rmse  = np.sqrt(np.mean((actual - predicted) ** 2))
    mape  = (np.mean(np.abs((actual[mask_nonzero] - predicted[mask_nonzero]) / actual[mask_nonzero])) * 100
             if mask_nonzero.sum() > 0 else np.nan)
    # Weighted MAPE（对高销量 SKU 加权，更贴近业务）
    wmape = (np.sum(np.abs(actual - predicted)) / np.sum(actual) * 100
             if actual.sum() > 0 else np.nan)
    bias  = np.mean(predicted - actual)

    return {
        "MAE":   round(mae,  2),
        "RMSE":  round(rmse, 2),
        "MAPE":  round(mape, 2) if not np.isnan(mape) else None,
        "WMAPE": round(wmape, 2) if not np.isnan(wmape) else None,
        "Bias":  round(bias, 2),
        "n":     len(actual),
    }


def evaluate_forecast(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    对比预测与实际，输出分组评估指标

    Args:
        forecast_df: 含 date/country/category/sku_id/forecast
        actual_df:   含 date/country/category/sku_id/sales_qty
        group_cols:  汇总维度
    """
    if group_cols is None:
        group_cols = ["country", "category"]

    merged = forecast_df.merge(
        actual_df[["date", "country", "category", "sku_id", "sales_qty"]],
        on=["date", "country", "category", "sku_id"],
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame()

    rows = []
    for keys, grp in merged.groupby(group_cols):
        m = compute_metrics(grp["sales_qty"].values, grp["forecast"].values)
        row = {k: v for k, v in zip(group_cols, (keys if isinstance(keys, tuple) else [keys]))}
        row.update(m)
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("WMAPE")
    return result


def flag_anomalies(
    forecast_df: pd.DataFrame,
    threshold_upper: float = 3.0,
    threshold_lower: float = 0.3,
    forecast_col: str = "forecast",
) -> pd.DataFrame:
    """
    标记预测相对基线的异常（节假日/大促导致的突增/骤降）
    threshold_upper: 超过滚动均值 N 倍视为高峰
    threshold_lower: 低于滚动均值 N 倍视为低谷
    """
    df = forecast_df.copy()
    df = df.sort_values(["country", "sku_id", "date"])

    col = forecast_col if forecast_col in df.columns else "forecast"
    df["roll_baseline"] = (
        df.groupby(["country", "sku_id"])[col]
        .transform(lambda x: x.rolling(7, min_periods=1).mean().shift(1))
    )
    df["forecast_ratio"] = df[col] / (df["roll_baseline"] + 1)
    df["anomaly_flag"] = "normal"
    df.loc[df["forecast_ratio"] >= threshold_upper, "anomaly_flag"] = "spike"
    df.loc[df["forecast_ratio"] <= threshold_lower, "anomaly_flag"] = "dip"

    return df
