"""
LightGBM 需求预测模型
全局训练（所有国家+SKU共用一个模型），利用 country/category 特征区分
"""

import pandas as pd
import numpy as np
import warnings
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.features import build_feature_matrix, LGBM_FEATURE_COLS

warnings.filterwarnings("ignore")


def train_lgbm(
    df_features: pd.DataFrame,
    train_end: str = "2025-09-30",
    val_end:   str = "2025-11-30",
    feature_cols: list[str] | None = None,
    target_col: str = "sales_qty",
) -> dict:
    """
    训练全局 LightGBM 模型

    Returns:
        dict: model, feature_importance, val_metrics
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("请先安装: pip install lightgbm")

    if feature_cols is None:
        feature_cols = [c for c in LGBM_FEATURE_COLS if c in df_features.columns]

    df = df_features.dropna(subset=feature_cols).copy()

    train_mask = df["date"] <= pd.Timestamp(train_end)
    val_mask   = (df["date"] > pd.Timestamp(train_end)) & (df["date"] <= pd.Timestamp(val_end))

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]
    X_val   = df.loc[val_mask, feature_cols]
    y_val   = df.loc[val_mask, target_col]

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val,   label=y_val, reference=lgb_train)

    params = {
        "objective":        "tweedie",        # 适合销量（右偏、非负）
        "tweedie_variance_power": 1.2,
        "metric":           "rmse",
        "learning_rate":    0.05,
        "num_leaves":       127,
        "max_depth":        -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "reg_alpha":        0.1,
        "reg_lambda":       0.1,
        "n_jobs":           -1,
        "verbose":          -1,
        "seed":             42,
    }

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )

    # 验证集指标
    val_preds = model.predict(X_val)
    val_preds = np.maximum(0, val_preds)

    actual = y_val.values
    mask_nonzero = actual > 0
    mae   = np.mean(np.abs(actual - val_preds))
    mape  = (np.mean(np.abs((actual[mask_nonzero] - val_preds[mask_nonzero]) / actual[mask_nonzero])) * 100
             if mask_nonzero.sum() > 0 else float("nan"))
    wmape = (np.sum(np.abs(actual - val_preds)) / np.sum(actual) * 100
             if actual.sum() > 0 else float("nan"))
    rmse  = np.sqrt(np.mean((actual - val_preds) ** 2))

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "feature_importance": importance,
        "val_metrics": {"MAE": mae, "MAPE": mape, "WMAPE": wmape, "RMSE": rmse},
        "best_iteration": model.best_iteration,
    }


def predict_lgbm(
    model_dict: dict,
    df_features: pd.DataFrame,
    predict_start: str = "2025-12-01",
    forecast_days: int = 30,
) -> pd.DataFrame:
    """
    用训练好的 LightGBM 模型进行未来 forecast_days 天预测
    采用递归预测：每预测一步，将预测值填入滞后特征，继续预测下一步
    """
    model = model_dict["model"]
    feature_cols = model_dict["feature_cols"]

    predict_end = pd.Timestamp(predict_start) + pd.Timedelta(days=forecast_days - 1)
    pred_dates  = pd.date_range(predict_start, predict_end, freq="D")

    # 准备历史数据用于递归滞后
    history = df_features[df_features["date"] < pd.Timestamp(predict_start)].copy()
    history = history.sort_values(["country", "sku_id", "date"])

    results = []

    for (country, category, sku_id), hist_grp in history.groupby(["country", "category", "sku_id"]):
        hist_grp = hist_grp.sort_values("date").copy()
        sales_hist = hist_grp.set_index("date")["sales_qty"].to_dict()

        for pred_date in pred_dates:
            # 构建该行特征
            row = _build_single_row(
                pred_date, country, category, sku_id,
                sales_hist, hist_grp, feature_cols
            )
            if row is None:
                continue

            X = pd.DataFrame([row])[feature_cols]
            pred = float(model.predict(X)[0])
            pred = max(0, round(pred))
            sales_hist[pred_date] = pred

            results.append({
                "date":     pred_date,
                "country":  country,
                "category": category,
                "sku_id":   sku_id,
                "forecast": pred,
                "model":    "lgbm",
            })

    return pd.DataFrame(results)


def _build_single_row(
    pred_date: pd.Timestamp,
    country: str, category: str, sku_id: str,
    sales_hist: dict,
    hist_grp: pd.DataFrame,
    feature_cols: list[str],
) -> dict | None:
    """构建单个预测行的特征字典"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from data.country_config import COUNTRY_PROFILE, PLATFORM_PROMOS

    row = {}

    # 时间特征
    row["dayofweek"]    = pred_date.dayofweek
    row["dayofmonth"]   = pred_date.day
    row["month"]        = pred_date.month
    row["quarter"]      = pred_date.quarter
    row["year"]         = pred_date.year
    row["is_weekend"]   = int(pred_date.dayofweek >= 5)
    row["is_month_end"] = int(pred_date == pred_date + pd.offsets.MonthEnd(0))
    row["weekofyear"]   = pred_date.isocalendar().week
    row["month_sin"]    = np.sin(2 * np.pi * pred_date.month / 12)
    row["month_cos"]    = np.cos(2 * np.pi * pred_date.month / 12)
    row["dayofweek_sin"] = np.sin(2 * np.pi * pred_date.dayofweek / 7)
    row["dayofweek_cos"] = np.cos(2 * np.pi * pred_date.dayofweek / 7)

    # 滞后特征（从 sales_hist 回溯）
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        lag_date = pred_date - pd.Timedelta(days=lag)
        row[f"lag_{lag}d"] = sales_hist.get(lag_date, np.nan)

    # 滚动均值
    for window in [7, 14, 28]:
        vals = [sales_hist.get(pred_date - pd.Timedelta(days=i), np.nan) for i in range(1, window + 1)]
        vals = [v for v in vals if not np.isnan(v)]
        row[f"roll_mean_{window}d"] = np.mean(vals) if vals else np.nan
        row[f"roll_std_{window}d"]  = np.std(vals)  if len(vals) > 1 else 0
        row[f"roll_max_{window}d"]  = max(vals) if vals else np.nan

    row["lag_28d_trend"] = (row.get("roll_mean_7d", 0) - row.get("roll_mean_28d", 0)) / (row.get("roll_mean_28d", 1) + 1)

    # 节假日（使用 holiday_boost 字段估计，或取历史平均）
    if "holiday_boost" in hist_grp.columns:
        hist_hb = hist_grp.set_index("date")["holiday_boost"].to_dict()
        # 对未来日期：若在节假日窗口内取配置，否则=1.0
        hb = hist_hb.get(pred_date, 1.0)
    else:
        hb = 1.0

    row["holiday_active"]    = int(hb > 1.05)
    row["holiday_intensity"] = hb - 1.0

    # 大促
    is_promo = 0
    days_to_next = 999
    for year in [pred_date.year]:
        for month, day, name, duration, _ in PLATFORM_PROMOS:
            try:
                start = pd.Timestamp(year=year, month=month, day=day)
                end = start + pd.Timedelta(days=duration - 1)
                if start <= pred_date <= end:
                    is_promo = 1
                diff = (start - pred_date).days
                if 0 <= diff < days_to_next:
                    days_to_next = diff
            except ValueError:
                pass
    row["is_promo_day"] = is_promo
    row["days_to_next_promo"] = days_to_next

    # 国家特征
    profile = COUNTRY_PROFILE.get(country, {})
    row["price_sensitivity"]  = profile.get("price_sensitivity", 0.7)
    row["promo_sensitivity"]  = profile.get("promo_sensitivity", 1.5)
    row["base_demand_level"]  = profile.get("base_demand_multiplier", 1.0)
    row["weekend_boost_rate"] = profile.get("weekend_boost", 1.2)

    # 编码
    from src.features import LGBM_FEATURE_COLS
    cat_list = sorted(["Electronics", "Fashion", "Beauty", "Home", "Sports", "Toys"])
    country_list = sorted(["SG", "MY", "TH", "PH", "VN"])
    row["category_enc"] = cat_list.index(category) if category in cat_list else -1
    row["country_enc"]  = country_list.index(country) if country in country_list else -1

    # 填充 NaN
    for col in feature_cols:
        if col not in row:
            row[col] = 0

    return row


def save_model(model_dict: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(model_dict, f)
    print(f"模型已保存: {path}")


def load_model(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
