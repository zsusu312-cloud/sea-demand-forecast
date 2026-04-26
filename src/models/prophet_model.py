"""
Prophet 模型封装
支持各国节假日配置，输出带置信区间的预测
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.country_config import HOLIDAYS

warnings.filterwarnings("ignore")


def _build_prophet_holidays(country: str) -> pd.DataFrame:
    """将国家节假日配置转为 Prophet holidays DataFrame"""
    rows = []
    for h in HOLIDAYS.get(country, []):
        for hdate_str in h["dates"]:
            hdate = pd.Timestamp(hdate_str)
            rows.append({
                "holiday": h["name"],
                "ds": hdate,
                "lower_window": -h["pre_days"],
                "upper_window": h["post_days"],
            })
    if not rows:
        return pd.DataFrame(columns=["holiday", "ds", "lower_window", "upper_window"])
    return pd.DataFrame(rows)


def fit_prophet(
    train_df: pd.DataFrame,
    country: str,
    forecast_days: int = 30,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
) -> dict:
    """
    训练单条时间序列的 Prophet 模型

    Args:
        train_df: 含 'date'、'sales_qty' 列的 DataFrame
        country: 国家代码（用于匹配节假日配置）
        forecast_days: 预测天数

    Returns:
        dict with keys: forecast_df, model, mae, mape
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("请先安装: pip install prophet")

    df_prophet = train_df[["date", "sales_qty"]].rename(
        columns={"date": "ds", "sales_qty": "y"}
    ).copy()
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
    df_prophet["y"] = df_prophet["y"].clip(lower=0)

    holidays_df = _build_prophet_holidays(country)

    model = Prophet(
        holidays=holidays_df if len(holidays_df) > 0 else None,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        seasonality_mode="multiplicative",   # 节假日效应呈乘法性
        changepoint_prior_scale=0.05,
        holidays_prior_scale=15.0,           # 节假日强先验
        seasonality_prior_scale=10.0,
        interval_width=0.8,
    )
    model.fit(df_prophet, iter=200)

    future = model.make_future_dataframe(periods=forecast_days, freq="D")
    forecast = model.predict(future)

    # 预测结果后处理
    forecast["yhat"] = forecast["yhat"].clip(lower=0).round()
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0).round()
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0).round()

    forecast_period = forecast[forecast["ds"] > df_prophet["ds"].max()].copy()
    forecast_period = forecast_period[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
        columns={"ds": "date", "yhat": "forecast", "yhat_lower": "lower_80", "yhat_upper": "upper_80"}
    )

    return {
        "forecast_df": forecast_period,
        "full_forecast": forecast,
        "model": model,
        "holidays_df": holidays_df,
    }


def batch_prophet_forecast(
    df: pd.DataFrame,
    forecast_days: int = 30,
    train_end: str = "2025-11-30",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    批量对所有 country × SKU 运行 Prophet 预测

    Returns:
        forecast_all: 含 country/category/sku_id/date/forecast/lower_80/upper_80
    """
    results = []
    groups = df.groupby(["country", "category", "sku_id"])
    total = len(groups)

    for i, ((country, category, sku_id), grp) in enumerate(groups):
        train = grp[grp["date"] <= pd.Timestamp(train_end)].copy()
        if len(train) < 60:
            continue

        try:
            res = fit_prophet(train, country, forecast_days)
            fdf = res["forecast_df"].copy()
            fdf["country"]  = country
            fdf["category"] = category
            fdf["sku_id"]   = sku_id
            fdf["model"]    = "prophet"
            results.append(fdf)
        except Exception as e:
            if verbose:
                print(f"  ⚠ Prophet 失败 {country}-{sku_id}: {e}")

        if verbose and (i + 1) % 20 == 0:
            print(f"  Prophet 进度: {i+1}/{total}")

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)
