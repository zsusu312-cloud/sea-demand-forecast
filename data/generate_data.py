"""
模拟数据生成器
生成 2023-01-01 ~ 2025-12-31 的 SKU 级日销量数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.country_config import COUNTRY_PROFILE, CATEGORIES, HOLIDAYS, PLATFORM_PROMOS

OUTPUT_DIR = Path(__file__).parent.parent / "data"
START_DATE = "2023-01-01"
END_DATE   = "2025-12-31"
RANDOM_SEED = 42


def _build_holiday_boost_series(dates: pd.DatetimeIndex, country: str, category: str) -> np.ndarray:
    """给定国家+品类，生成每日节假日倍数（1.0=无效应）"""
    boost = np.ones(len(dates))
    date_to_idx = {d: i for i, d in enumerate(dates)}

    for h in HOLIDAYS.get(country, []):
        for hdate_str in h["dates"]:
            hdate = pd.Timestamp(hdate_str)
            cat_boost = h["boost_by_category"].get(category, h["peak_boost"])
            pre  = h["pre_days"]
            post = h["post_days"]

            for offset in range(-pre, post + 1):
                d = hdate + pd.Timedelta(days=offset)
                if d in date_to_idx:
                    idx = date_to_idx[d]
                    # 越靠近节日倍数越高（线性衰减到边缘的50%）
                    if offset <= 0:
                        ratio = 1 - (abs(offset) / (pre + 1)) * 0.5 if pre > 0 else 1.0
                    else:
                        ratio = 1 - (offset / (post + 1)) * 0.5 if post > 0 else 1.0
                    this_boost = 1 + (cat_boost - 1) * ratio
                    boost[idx] = max(boost[idx], this_boost)  # 节日叠加取最大

    return boost


def _build_promo_boost_series(dates: pd.DatetimeIndex, promo_sensitivity: float) -> np.ndarray:
    """平台大促效应"""
    boost = np.ones(len(dates))
    for year in [2023, 2024, 2025]:
        for month, day, name, duration, base_boost in PLATFORM_PROMOS:
            try:
                start = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                continue
            # 提前3天预热
            for offset in range(-3, duration + 1):
                d = start + pd.Timedelta(days=offset)
                if d in dates:
                    idx = dates.get_loc(d)
                    if offset < 0:
                        ratio = 0.4 + 0.6 * (1 - abs(offset) / 3)
                    else:
                        ratio = 1.0
                    this_boost = 1 + (base_boost - 1) * ratio * promo_sensitivity
                    boost[idx] = max(boost[idx], this_boost)
    return boost


def _weekly_pattern(dayofweek: int, country: str) -> float:
    """各国周内消费模式差异"""
    # 0=Mon, 6=Sun
    patterns = {
        "SG": [0.85, 0.90, 0.95, 1.00, 1.10, 1.25, 1.20],
        "MY": [0.80, 0.85, 0.90, 0.95, 1.15, 1.30, 1.25],
        "TH": [0.85, 0.88, 0.92, 1.00, 1.08, 1.20, 1.18],
        "PH": [0.80, 0.82, 0.88, 0.95, 1.10, 1.35, 1.40],
        "VN": [0.90, 0.92, 0.95, 1.00, 1.05, 1.12, 1.08],
    }
    return patterns.get(country, [1.0] * 7)[dayofweek]


def _annual_trend(dates: pd.DatetimeIndex, growth_rate: float = 0.18) -> np.ndarray:
    """年增长趋势（东南亚电商年均~18%增速）"""
    years_elapsed = (dates - dates[0]).days / 365.25
    return 1 + growth_rate * years_elapsed


def _monthly_seasonality(month: int, category: str) -> float:
    """月度季节性（不含节假日）"""
    patterns = {
        "Electronics": [0.90, 0.75, 0.85, 0.90, 0.95, 0.95,
                        1.00, 1.05, 1.05, 1.10, 1.15, 1.35],
        "Fashion":     [0.85, 0.80, 0.90, 1.05, 1.00, 0.95,
                        1.00, 1.00, 1.05, 1.05, 1.10, 1.25],
        "Beauty":      [0.90, 0.85, 1.05, 1.10, 1.05, 0.95,
                        0.90, 0.95, 1.00, 1.05, 1.10, 1.10],
        "Home":        [0.95, 0.80, 0.90, 0.95, 1.00, 1.00,
                        1.00, 1.00, 1.00, 1.00, 1.05, 1.35],
        "Sports":      [1.20, 1.10, 1.10, 1.05, 1.00, 0.90,
                        0.90, 0.95, 1.00, 1.00, 0.95, 0.85],
        "Toys":        [0.70, 0.65, 0.75, 0.80, 0.85, 0.90,
                        0.95, 0.95, 1.00, 1.10, 1.30, 1.65],
    }
    return patterns.get(category, [1.0] * 12)[month - 1]


def generate_sku_timeseries(
    country: str,
    category: str,
    sku: str,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    profile = COUNTRY_PROFILE[country]
    cat_cfg  = CATEGORIES[category]

    base = cat_cfg["avg_daily_base"] * profile["base_demand_multiplier"]
    # SKU 个体差异：同品类内有强/弱 SKU
    sku_multiplier = rng.uniform(0.3, 1.8)

    n = len(dates)

    # --- 各效应叠加 ---
    trend    = _annual_trend(dates)
    monthly  = np.array([_monthly_seasonality(d.month, category) for d in dates])
    weekly   = np.array([_weekly_pattern(d.dayofweek, country) for d in dates])
    holiday  = _build_holiday_boost_series(dates, country, category)
    promo    = _build_promo_boost_series(dates, profile["promo_sensitivity"])

    # 基础量（显式转为 numpy array，避免 pandas Index 问题）
    demand_float = np.array(base * sku_multiplier * trend * monthly * weekly * holiday * promo, dtype=float)

    # 加噪（泊松 + 长尾偶发大单）
    noise = rng.normal(1.0, 0.12, n)
    demand_float = demand_float * np.clip(noise, 0.5, 2.5)

    # 偶发促销外溢（每季度1-2次随机spike）
    spike_days = rng.choice(n, size=int(n * 0.015), replace=False)
    demand_float[spike_days] *= rng.uniform(1.5, 3.0, len(spike_days))

    # 取整，保证非负
    demand = np.maximum(0, np.round(demand_float)).astype(int)

    # 缺货模拟：随机5%天数缺货（需求记为0，但真实需求存档）
    stockout_mask = rng.random(n) < 0.05
    observed = demand.copy()
    observed[stockout_mask] = 0

    df = pd.DataFrame({
        "date": dates,
        "country": country,
        "category": category,
        "sku_id": sku,
        "sales_qty": observed,
        "true_demand": demand,     # 含缺货期真实需求（实际中不可见，仅用于评估）
        "is_stockout": stockout_mask.astype(int),
        "holiday_boost": np.round(holiday, 4),
        "promo_boost": np.round(promo, 4),
    })
    return df


def generate_all(output_dir: Path = OUTPUT_DIR):
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    all_dfs = []
    total = sum(len(v["skus"]) for v in CATEGORIES.values()) * len(COUNTRY_PROFILE)
    done = 0

    for country in COUNTRY_PROFILE:
        for category, cat_cfg in CATEGORIES.items():
            for sku in cat_cfg["skus"]:
                df = generate_sku_timeseries(country, category, sku, dates, rng)
                all_dfs.append(df)
                done += 1
                if done % 10 == 0:
                    print(f"  生成进度: {done}/{total} ({country}-{category}-{sku})")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sales_history.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\n[OK] 数据生成完成: {len(combined):,} 行 -> {out_path}")
    print(f"   时间范围: {combined.date.min().date()} ~ {combined.date.max().date()}")
    print(f"   SKU总数: {combined.sku_id.nunique()} 个 x {combined.country.nunique()} 国")
    print(f"   文件大小: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # 同时存一份 CSV 方便查看
    sample = combined.sample(5000, random_state=42).sort_values(["country", "date"])
    sample.to_csv(output_dir / "sales_sample_5k.csv", index=False)
    print(f"   CSV样本(5k行): {output_dir / 'sales_sample_5k.csv'}")

    return combined


if __name__ == "__main__":
    print("开始生成东南亚 SKU 级模拟销售数据...")
    generate_all()
