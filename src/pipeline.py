"""
主流水线：数据生成 → 特征工程 → 模型训练 → 预测 → 结果保存
运行: python src/pipeline.py
"""

import sys
import pandas as pd
import numpy as np
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))   # 确保项目根目录在 import 路径中

DATA_DIR   = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def run(
    regenerate_data: bool = False,
    run_prophet:     bool = True,
    run_lgbm:        bool = True,
    forecast_days:   int  = 30,
    train_end:       str  = "2025-10-31",
    val_end:         str  = "2025-11-30",
    predict_start:   str  = "2025-12-01",
    verbose:         bool = True,
):
    t0 = time.time()

    # ── 1. 数据 ──────────────────────────────────────────────
    sales_path = DATA_DIR / "sales_history.parquet"

    if regenerate_data or not sales_path.exists():
        print("=" * 50)
        print("步骤 1/4: 生成模拟数据...")
        from data.generate_data import generate_all
        df = generate_all(DATA_DIR)
    else:
        print("步骤 1/4: 加载已有数据...")
        df = pd.read_parquet(sales_path)
        df["date"] = pd.to_datetime(df["date"])
        print(f"  已加载 {len(df):,} 行数据")

    # ── 2. 特征工程（LightGBM 用）─────────────────────────────
    print("\n步骤 2/4: 特征工程...")
    feat_path = OUTPUT_DIR / "features.parquet"

    if not feat_path.exists() or regenerate_data:
        from src.features import build_feature_matrix
        df_feat = build_feature_matrix(df)
        df_feat.to_parquet(feat_path, index=False)
        print(f"  特征矩阵已保存: {feat_path} ({len(df_feat.columns)} 列)")
    else:
        df_feat = pd.read_parquet(feat_path)
        df_feat["date"] = pd.to_datetime(df_feat["date"])
        print(f"  加载已有特征矩阵: {len(df_feat.columns)} 列")

    # ── 3. 模型预测 ────────────────────────────────────────────
    print("\n步骤 3/4: 模型训练与预测...")
    all_forecasts = []

    if run_prophet:
        print("  [Prophet] 批量预测中（按 SKU 训练）...")
        from src.models.prophet_model import batch_prophet_forecast
        prophet_fc = batch_prophet_forecast(
            df,
            forecast_days=forecast_days,
            train_end=val_end,
            verbose=verbose,
        )
        if not prophet_fc.empty:
            prophet_fc.to_parquet(OUTPUT_DIR / "forecast_prophet.parquet", index=False)
            all_forecasts.append(prophet_fc)
            print(f"  [OK] Prophet 完成: {len(prophet_fc)} 行预测")

    if run_lgbm:
        print("  [LightGBM] 全局模型训练中...")
        from src.models.lgbm_model import train_lgbm, predict_lgbm, save_model

        result = train_lgbm(df_feat, train_end=train_end, val_end=val_end)
        model_dict = result
        print(f"  [OK] LightGBM 训练完成 | 验证集 MAPE: {result['val_metrics']['MAPE']:.1f}% "
              f"| WMAPE: {result['val_metrics']['WMAPE']:.1f}%  (best iter: {result['best_iteration']})")

        save_model(model_dict, str(OUTPUT_DIR / "lgbm_model.pkl"))

        print("  [LightGBM] 未来预测中（递归方式）...")
        lgbm_fc = predict_lgbm(model_dict, df_feat, predict_start=predict_start, forecast_days=forecast_days)
        if not lgbm_fc.empty:
            lgbm_fc.to_parquet(OUTPUT_DIR / "forecast_lgbm.parquet", index=False)
            all_forecasts.append(lgbm_fc)
            print(f"  [OK] LightGBM 完成: {len(lgbm_fc)} 行预测")

    # ── 4. 合并 + 集成 ─────────────────────────────────────────
    print("\n步骤 4/4: 合并预测结果...")

    if all_forecasts:
        if len(all_forecasts) >= 2:
            # 简单集成：Prophet × 0.4 + LightGBM × 0.6
            prophet_part = all_forecasts[0][["date", "country", "category", "sku_id", "forecast"]].copy()
            lgbm_part    = all_forecasts[1][["date", "country", "category", "sku_id", "forecast"]].copy()
            prophet_part = prophet_part.rename(columns={"forecast": "prophet_fc"})
            lgbm_part    = lgbm_part.rename(columns={"forecast": "lgbm_fc"})
            ensemble = prophet_part.merge(lgbm_part, on=["date", "country", "category", "sku_id"], how="outer")
            ensemble["forecast_ensemble"] = (
                ensemble["prophet_fc"].fillna(0) * 0.4 +
                ensemble["lgbm_fc"].fillna(0) * 0.6
            ).round().astype(int)
            ensemble["final_forecast"] = ensemble["forecast_ensemble"]
        else:
            ensemble = all_forecasts[0].copy()
            ensemble["final_forecast"] = ensemble["forecast"]

        # 附加节假日元数据
        from src.evaluate import flag_anomalies
        ensemble = flag_anomalies(ensemble, forecast_col="final_forecast"
                                  if "final_forecast" in ensemble.columns
                                  else "forecast")

        out_path = OUTPUT_DIR / "forecast_final.parquet"
        ensemble.to_parquet(out_path, index=False)

        # 同时输出 Excel 汇总（按国家×品类×日期汇总）
        summary = (
            ensemble.groupby(["country", "category", "date"])
            [["final_forecast"] if "final_forecast" in ensemble.columns else ["forecast"]]
            .sum()
            .reset_index()
        )
        summary.to_csv(OUTPUT_DIR / "forecast_summary.csv", index=False)

        print(f"\n[DONE] 流水线完成！耗时 {time.time()-t0:.0f}s")
        print(f"   最终预测: {out_path}")
        print(f"   汇总 CSV: {OUTPUT_DIR / 'forecast_summary.csv'}")
        print(f"\n  预测期间: {ensemble['date'].min().date()} ~ {ensemble['date'].max().date()}")
        print(f"  覆盖国家: {sorted(ensemble['country'].unique())}")
        print(f"  覆盖 SKU: {ensemble['sku_id'].nunique()} 个")

        return ensemble
    else:
        print("[WARN] 无预测结果，请检查模型配置")
        return pd.DataFrame()


def flag_anomalies(df: pd.DataFrame, forecast_col: str = "final_forecast") -> pd.DataFrame:
    """标记预测峰值/低谷"""
    df = df.copy()
    df = df.sort_values(["country", "sku_id", "date"])

    if forecast_col in df.columns:
        df["roll_baseline"] = (
            df.groupby(["country", "sku_id"])[forecast_col]
            .transform(lambda x: x.expanding(min_periods=1).mean())
        )
        df["forecast_ratio"] = df[forecast_col] / (df["roll_baseline"] + 1)
        df["anomaly_flag"] = "normal"
        df.loc[df["forecast_ratio"] >= 2.5, "anomaly_flag"] = "spike"
        df.loc[df["forecast_ratio"] <= 0.4, "anomaly_flag"] = "dip"

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEA 需求预测流水线")
    parser.add_argument("--regenerate", action="store_true", help="重新生成模拟数据")
    parser.add_argument("--no-prophet", action="store_true", help="跳过 Prophet")
    parser.add_argument("--no-lgbm",   action="store_true", help="跳过 LightGBM")
    parser.add_argument("--days",      type=int, default=30, help="预测天数")
    args = parser.parse_args()

    run(
        regenerate_data=args.regenerate,
        run_prophet=not args.no_prophet,
        run_lgbm=not args.no_lgbm,
        forecast_days=args.days,
    )
