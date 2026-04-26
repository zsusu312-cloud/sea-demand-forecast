"""
东南亚 SKU 需求预测看板
运行: streamlit run dashboard/app.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── 页面配置 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="SEA 需求预测 | 速卖通供应链",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

COUNTRY_FLAG = {"SG": "🇸🇬", "MY": "🇲🇾", "TH": "🇹🇭", "PH": "🇵🇭", "VN": "🇻🇳"}
COUNTRY_NAME = {"SG": "新加坡", "MY": "马来西亚", "TH": "泰国", "PH": "菲律宾", "VN": "越南"}
CATEGORY_ICON = {
    "Electronics": "💻", "Fashion": "👗", "Beauty": "💄",
    "Home": "🏠", "Sports": "🏃", "Toys": "🧸",
}

# ── 数据加载 ──────────────────────────────────────────────────
def load_sales() -> pd.DataFrame:
    # session_state 兜底（云端无法写磁盘时使用）
    if "_sales_df" in st.session_state:
        return st.session_state["_sales_df"]
    path = ROOT / "data" / "sales_history.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_forecast() -> pd.DataFrame:
    if "_forecast_df" in st.session_state:
        return st.session_state["_forecast_df"]
    path = ROOT / "output" / "forecast_final.parquet"
    if not path.exists():
        for name in ["forecast_prophet", "forecast_lgbm"]:
            p = ROOT / "output" / f"{name}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"])
                df["final_forecast"] = df.get("forecast", df.get("yhat", 0))
                return df
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    if "final_forecast" not in df.columns:
        df["final_forecast"] = df.get("forecast", df.get("yhat", 0))
    return df


@st.cache_data(show_spinner=False)
def generate_data_cached() -> pd.DataFrame:
    """云端友好：数据在内存中生成，不写磁盘"""
    from data.generate_data import generate_all
    import tempfile, os
    tmp = Path(tempfile.mkdtemp())
    df = generate_all(tmp)
    return df


@st.cache_data(show_spinner=False)
def generate_forecast_cached(df_hash: int, _df: pd.DataFrame) -> pd.DataFrame:
    """基线预测，纯内存，不写磁盘"""
    groups = _df.groupby(["country", "category", "sku_id"])
    results = []
    pred_dates = pd.date_range("2025-12-01", periods=30, freq="D")
    for (country, category, sku_id), grp in groups:
        grp = grp.sort_values("date")
        last_28 = grp.tail(28)["sales_qty"].values
        if len(last_28) == 0:
            continue
        mean_val = np.mean(last_28)
        recent_mean = np.mean(last_28[-7:]) if len(last_28) >= 7 else mean_val
        trend = (recent_mean - mean_val) / (mean_val + 1) * 0.3
        for i, d in enumerate(pred_dates):
            day_boost = 1.2 if d.dayofweek >= 5 else 1.0
            val = max(0, round(mean_val * (1 + trend * i / 30) * day_boost))
            results.append({
                "date": d, "country": country, "category": category,
                "sku_id": sku_id, "forecast": val, "final_forecast": val,
                "model": "baseline_ma",
            })
    return pd.DataFrame(results)


def generate_data_inline():
    """触发数据生成（云端：内存生成；本地：优先读文件）"""
    with st.spinner("正在生成模拟数据（约 20-40s）..."):
        path = ROOT / "data" / "sales_history.parquet"
        if not path.exists():
            df = generate_data_cached()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(path, index=False)
            except Exception:
                st.session_state["_sales_df"] = df
    st.cache_data.clear()
    st.success("数据生成完成！")
    st.rerun()


def run_quick_forecast(df: pd.DataFrame):
    """快速基线预测"""
    with st.spinner("运行快速基线预测（移动平均）..."):
        fc_df = generate_forecast_cached(hash(str(df.shape)), df)
        try:
            out = ROOT / "output" / "forecast_final.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            fc_df.to_parquet(out, index=False)
        except Exception:
            st.session_state["_forecast_df"] = fc_df
    st.cache_data.clear()
    st.success("基线预测完成！")
    st.rerun()


# ── 主页面 ────────────────────────────────────────────────────
def main():
    # 顶部标题栏
    col_title, col_meta = st.columns([3, 1])
    with col_title:
        st.markdown("## 📦 东南亚需求预测看板")
        st.caption("速卖通计划供应链 | SG · MY · TH · PH · VN | SKU 级 · 未来30天")
    with col_meta:
        st.markdown("")
        if st.button("🔄 刷新数据缓存"):
            st.cache_data.clear()
            st.rerun()

    df_sales    = load_sales()
    df_forecast = load_forecast()

    # ── 数据准备引导 ──────────────────────────────────────────
    if df_sales.empty:
        st.warning("⚠ 未检测到历史数据，请先生成模拟数据")
        if st.button("🚀 一键生成模拟数据（含节假日效应）", type="primary"):
            generate_data_inline()
        st.stop()

    if df_forecast.empty:
        st.info("📊 已有历史数据，尚未运行预测模型")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚡ 快速基线预测（秒级）", type="primary"):
                run_quick_forecast(df_sales)
        with col2:
            st.code("python src/pipeline.py", language="bash")
            st.caption("运行完整 Prophet + LightGBM 管道")
        st.stop()

    # ── 侧边栏筛选器 ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎛 筛选条件")

        countries = sorted(df_sales["country"].unique())
        sel_countries = st.multiselect(
            "国家",
            options=countries,
            default=countries,
            format_func=lambda c: f"{COUNTRY_FLAG.get(c,'')} {COUNTRY_NAME.get(c, c)}",
        )

        categories = sorted(df_sales["category"].unique())
        sel_categories = st.multiselect(
            "品类",
            options=categories,
            default=categories,
            format_func=lambda c: f"{CATEGORY_ICON.get(c,'')} {c}",
        )

        all_skus = sorted(df_sales[
            df_sales["country"].isin(sel_countries) &
            df_sales["category"].isin(sel_categories)
        ]["sku_id"].unique())

        sel_sku = st.selectbox("SKU（详细视图）", options=["(全部汇总)"] + all_skus)

        st.markdown("---")
        st.markdown("### 📅 历史回溯窗口")
        lookback_days = st.slider("回溯天数", 30, 365, 90)

        st.markdown("---")
        st.markdown("### ⚙ 预测设置")
        show_confidence = st.checkbox("显示置信区间", value=True)
        agg_level = st.radio("汇总粒度", ["国家", "品类", "SKU"], index=0)

    if not sel_countries or not sel_categories:
        st.warning("请至少选择一个国家和品类")
        st.stop()

    # 过滤数据
    sales_filt = df_sales[
        df_sales["country"].isin(sel_countries) &
        df_sales["category"].isin(sel_categories)
    ].copy()

    fc_filt = df_forecast[
        df_forecast["country"].isin(sel_countries) &
        df_forecast["category"].isin(sel_categories)
    ].copy()

    if sel_sku != "(全部汇总)":
        sales_filt = sales_filt[sales_filt["sku_id"] == sel_sku]
        fc_filt    = fc_filt[fc_filt["sku_id"] == sel_sku]

    # ── KPI 卡片 ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 预测期 KPI 总览（未来30天）")

    fc_col = "final_forecast" if "final_forecast" in fc_filt.columns else "forecast"

    total_fc  = fc_filt[fc_col].sum()
    total_hist = sales_filt[sales_filt["date"] >= sales_filt["date"].max() - pd.Timedelta(days=30)]["sales_qty"].sum()
    yoy_change = (total_fc - total_hist) / (total_hist + 1) * 100

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.metric("预测总需求量", f"{total_fc:,.0f}", f"{yoy_change:+.1f}% vs 近30天")
    with k2:
        top_country = fc_filt.groupby("country")[fc_col].sum().idxmax() if not fc_filt.empty else "-"
        top_country_val = fc_filt.groupby("country")[fc_col].sum().max() if not fc_filt.empty else 0
        st.metric(
            "需求最大国家",
            f"{COUNTRY_FLAG.get(top_country,'')} {top_country}",
            f"{top_country_val:,.0f} 件",
        )
    with k3:
        top_cat = fc_filt.groupby("category")[fc_col].sum().idxmax() if not fc_filt.empty else "-"
        top_cat_val = fc_filt.groupby("category")[fc_col].sum().max() if not fc_filt.empty else 0
        st.metric(
            "需求最大品类",
            f"{CATEGORY_ICON.get(top_cat,'')} {top_cat}",
            f"{top_cat_val:,.0f} 件",
        )
    with k4:
        peak_day = fc_filt.groupby("date")[fc_col].sum().idxmax() if not fc_filt.empty else "-"
        peak_val = fc_filt.groupby("date")[fc_col].sum().max() if not fc_filt.empty else 0
        st.metric("预测峰值日", str(peak_day)[:10] if peak_day != "-" else "-", f"{peak_val:,.0f} 件/天")
    with k5:
        avg_daily = fc_filt.groupby("date")[fc_col].sum().mean() if not fc_filt.empty else 0
        st.metric("日均需求量", f"{avg_daily:,.0f} 件/天")

    # ── 主图：历史 + 预测趋势 ─────────────────────────────────
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 趋势预测", "🌍 国家对比", "📦 品类分析", "🎉 节假日洞察", "🏆 SKU 排行"
    ])

    with tab1:
        _render_trend_tab(sales_filt, fc_filt, fc_col, lookback_days, show_confidence, agg_level, sel_countries, sel_categories)

    with tab2:
        _render_country_tab(sales_filt, fc_filt, fc_col, sel_countries)

    with tab3:
        _render_category_tab(sales_filt, fc_filt, fc_col, sel_categories)

    with tab4:
        _render_holiday_tab(fc_filt, fc_col)

    with tab5:
        _render_sku_ranking_tab(fc_filt, fc_col, sel_countries, sel_categories)


# ── Tab 渲染函数 ──────────────────────────────────────────────

def _render_trend_tab(sales, fc, fc_col, lookback_days, show_ci, agg_level, countries, categories):
    st.markdown("##### 历史销量 vs 预测趋势")

    cutoff = sales["date"].max() - pd.Timedelta(days=lookback_days)
    hist_agg = sales[sales["date"] >= cutoff].groupby("date")["sales_qty"].sum().reset_index()
    fc_agg   = fc.groupby("date")[fc_col].sum().reset_index().rename(columns={fc_col: "forecast"})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_agg["date"], y=hist_agg["sales_qty"],
        mode="lines", name="历史销量",
        line=dict(color="#4C78A8", width=2),
        hovertemplate="历史: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=fc_agg["date"], y=fc_agg["forecast"],
        mode="lines+markers", name="预测需求",
        line=dict(color="#F58518", width=2.5, dash="dash"),
        marker=dict(size=5),
        hovertemplate="预测: %{y:,.0f}<extra></extra>",
    ))

    # 置信区间（如有）
    if show_ci and "upper_80" in fc.columns and "lower_80" in fc.columns:
        upper = fc.groupby("date")["upper_80"].sum().reset_index()
        lower = fc.groupby("date")["lower_80"].sum().reset_index()
        fig.add_trace(go.Scatter(
            x=pd.concat([upper["date"], lower["date"].iloc[::-1]]),
            y=pd.concat([upper["upper_80"], lower["lower_80"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(245,133,24,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% 置信区间",
        ))

    # 标注预测起始线
    if not fc.empty:
        fc_start = fc["date"].min()
        fig.add_vline(x=fc_start.timestamp() * 1000, line_dash="dot", line_color="gray")
        fig.add_annotation(x=fc_start, y=1, yref="paper", text="预测起点",
                           showarrow=False, font=dict(color="gray", size=11),
                           xanchor="left")

    fig.update_layout(
        height=420, hovermode="x unified",
        xaxis_title="日期", yaxis_title="销售量（件）",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 数据表
    with st.expander("📋 预测明细数据"):
        display_cols = ["date", "country", "category", "sku_id", fc_col]
        if "anomaly_flag" in fc.columns:
            display_cols.append("anomaly_flag")
        st.dataframe(
            fc[display_cols].sort_values(["country", "date"]).head(500),
            use_container_width=True,
            height=300,
        )


def _render_country_tab(sales, fc, fc_col, countries):
    st.markdown("##### 各国需求分布对比")

    col1, col2 = st.columns(2)

    with col1:
        # 各国预测总量
        country_fc = fc.groupby("country")[fc_col].sum().reset_index()
        country_fc["country_label"] = country_fc["country"].map(
            lambda c: f"{COUNTRY_FLAG.get(c,'')} {COUNTRY_NAME.get(c, c)}"
        )
        fig = px.bar(
            country_fc.sort_values(fc_col, ascending=True),
            x=fc_col, y="country_label", orientation="h",
            color=fc_col, color_continuous_scale="Blues",
            title="预测30天总需求量（件）",
            labels={fc_col: "需求量", "country_label": "国家"},
        )
        fig.update_layout(height=320, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 各国日均趋势折线
        daily_country = fc.groupby(["date", "country"])[fc_col].sum().reset_index()
        fig2 = px.line(
            daily_country, x="date", y=fc_col, color="country",
            title="各国日需求趋势（预测期）",
            labels={fc_col: "需求量", "date": "日期", "country": "国家"},
            color_discrete_map={
                "SG": "#2196F3", "MY": "#4CAF50", "TH": "#FF9800",
                "PH": "#9C27B0", "VN": "#F44336",
            },
        )
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # 热力图：国家 × 品类需求矩阵
    st.markdown("##### 国家 × 品类 需求热力图（预测期总量）")
    pivot = fc.groupby(["country", "category"])[fc_col].sum().reset_index().pivot(
        index="country", columns="category", values=fc_col
    ).fillna(0)

    fig3 = px.imshow(
        pivot.astype(int),
        text_auto=True, aspect="auto",
        color_continuous_scale="YlOrRd",
        labels=dict(x="品类", y="国家", color="需求量"),
        title="需求热力图（30天预测总量）",
    )
    fig3.update_layout(height=320)
    st.plotly_chart(fig3, use_container_width=True)

    # 各国环比增长
    st.markdown("##### 各国近期趋势分析")
    hist_all = load_sales()
    if not hist_all.empty:
        metrics_rows = []
        for country in countries:
            c_sales = hist_all[hist_all["country"] == country].groupby("date")["sales_qty"].sum()
            c_fc    = fc[fc["country"] == country].groupby("date")[fc_col].sum()
            if len(c_sales) < 60:
                continue
            last30  = c_sales.tail(30).sum()
            prev30  = c_sales.iloc[-60:-30].sum()
            fc_next = c_fc.sum()
            mom     = (last30 - prev30) / (prev30 + 1) * 100
            fc_vs_hist = (fc_next - last30) / (last30 + 1) * 100
            metrics_rows.append({
                "国家": f"{COUNTRY_FLAG.get(country,'')} {COUNTRY_NAME.get(country,country)}",
                "上月销量": f"{prev30:,.0f}",
                "近30天销量": f"{last30:,.0f}",
                "环比增长": f"{mom:+.1f}%",
                "预测30天": f"{fc_next:,.0f}",
                "预测 vs 历史": f"{fc_vs_hist:+.1f}%",
            })
        if metrics_rows:
            st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)


def _render_category_tab(sales, fc, fc_col, categories):
    st.markdown("##### 品类需求分析")

    col1, col2 = st.columns(2)

    with col1:
        cat_fc = fc.groupby("category")[fc_col].sum().reset_index()
        cat_fc["icon_label"] = cat_fc["category"].map(
            lambda c: f"{CATEGORY_ICON.get(c,'')} {c}"
        )
        fig = px.pie(
            cat_fc, values=fc_col, names="icon_label",
            title="品类需求占比（预测期）",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 品类 × 国家分组柱状图
        cat_country = fc.groupby(["category", "country"])[fc_col].sum().reset_index()
        fig2 = px.bar(
            cat_country, x="category", y=fc_col, color="country",
            title="各品类国家需求分解",
            labels={fc_col: "需求量", "category": "品类", "country": "国家"},
            barmode="stack",
            color_discrete_map={
                "SG": "#2196F3", "MY": "#4CAF50", "TH": "#FF9800",
                "PH": "#9C27B0", "VN": "#F44336",
            },
        )
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # 品类趋势
    st.markdown("##### 品类日需求趋势（预测期）")
    daily_cat = fc.groupby(["date", "category"])[fc_col].sum().reset_index()
    fig3 = px.line(
        daily_cat, x="date", y=fc_col, color="category",
        facet_col="category", facet_col_wrap=3,
        labels={fc_col: "需求量", "date": "日期"},
        height=480,
    )
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)


def _render_holiday_tab(fc, fc_col):
    st.markdown("##### 节假日需求冲击分析")

    from data.country_config import HOLIDAYS, PLATFORM_PROMOS

    if fc.empty:
        st.info("暂无预测数据")
        return

    fc_dates = fc["date"].unique()
    fc_start = min(fc_dates)
    fc_end   = max(fc_dates)

    # 收集预测期内的节假日
    holiday_events = []
    for country, holidays in HOLIDAYS.items():
        for h in holidays:
            for hdate_str in h["dates"]:
                hdate = pd.Timestamp(hdate_str)
                if fc_start <= hdate <= fc_end:
                    holiday_events.append({
                        "国家": f"{COUNTRY_FLAG.get(country,'')} {country}",
                        "节假日": h["name"].replace("_", " "),
                        "日期": hdate.date(),
                        "提前备货天数": h["pre_days"],
                        "高峰倍数": h.get("peak_boost", 1.0),
                    })

    # 平台大促
    for year in [fc_start.year, fc_end.year]:
        for month, day, name, duration, boost in PLATFORM_PROMOS:
            try:
                promo_date = pd.Timestamp(year=year, month=month, day=day)
                if fc_start <= promo_date <= fc_end:
                    holiday_events.append({
                        "国家": "🌐 全区",
                        "节假日": name,
                        "日期": promo_date.date(),
                        "提前备货天数": 7,
                        "高峰倍数": boost,
                    })
            except ValueError:
                pass

    if holiday_events:
        st.markdown("**预测期内重要节点**")
        events_df = pd.DataFrame(holiday_events).sort_values("日期")
        st.dataframe(events_df, use_container_width=True, hide_index=True)
    else:
        st.info("预测期内无重大节假日（当前预测窗口：12月）")

    st.markdown("---")
    st.markdown("**各国节假日影响系数说明**")

    country_tabs = st.tabs([f"{COUNTRY_FLAG.get(c,'')} {c}" for c in ["SG", "MY", "TH", "PH", "VN"]])

    for i, country in enumerate(["SG", "MY", "TH", "PH", "VN"]):
        with country_tabs[i]:
            h_list = HOLIDAYS.get(country, [])
            rows = []
            for h in h_list:
                for cat, boost in h["boost_by_category"].items():
                    rows.append({
                        "节假日": h["name"].replace("_", " "),
                        "品类": f"{CATEGORY_ICON.get(cat,'')} {cat}",
                        "需求倍数": boost,
                        "提前备货天数": h["pre_days"],
                    })
            if rows:
                df_h = pd.DataFrame(rows)
                pivot = df_h.pivot(index="节假日", columns="品类", values="需求倍数").fillna(1.0)
                fig = px.imshow(
                    pivot, text_auto=".1f", aspect="auto",
                    color_continuous_scale=[(0, "#d73027"), (0.5, "#ffffbf"), (1, "#1a9850")],
                    zmin=0.3, zmax=4.0,
                    title=f"{COUNTRY_NAME.get(country, country)} 节假日 × 品类 需求倍数",
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

                from data.country_config import COUNTRY_PROFILE
                profile = COUNTRY_PROFILE.get(country, {})
                col1, col2, col3 = st.columns(3)
                col1.metric("促销敏感度", f"{profile.get('promo_sensitivity', 1.0):.1f}x")
                col2.metric("价格敏感度", f"{profile.get('price_sensitivity', 0.7):.0%}")
                col3.metric("周末消费增幅", f"+{(profile.get('weekend_boost', 1.2)-1)*100:.0f}%")


def _render_sku_ranking_tab(fc, fc_col, countries, categories):
    st.markdown("##### SKU 需求排行与异常预警")

    if fc.empty:
        st.info("暂无预测数据")
        return

    sku_agg = fc.groupby(["country", "category", "sku_id"]).agg(
        total_forecast=(fc_col, "sum"),
        avg_daily=(fc_col, "mean"),
        peak_day=(fc_col, "max"),
    ).reset_index().sort_values("total_forecast", ascending=False)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Top 20 高需求 SKU**")
        top20 = sku_agg.head(20).copy()
        top20["country_flag"] = top20["country"].map(COUNTRY_FLAG)
        top20["cat_icon"]     = top20["category"].map(CATEGORY_ICON)
        top20["display"] = top20["country_flag"] + " " + top20["cat_icon"] + " " + top20["sku_id"]

        fig = px.bar(
            top20.sort_values("total_forecast"),
            x="total_forecast", y="display", orientation="h",
            color="country",
            color_discrete_map={
                "SG": "#2196F3", "MY": "#4CAF50", "TH": "#FF9800",
                "PH": "#9C27B0", "VN": "#F44336",
            },
            labels={"total_forecast": "30天预测总量", "display": "SKU"},
            height=520,
        )
        fig.update_layout(showlegend=True, yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**📊 SKU 汇总统计**")
        display_df = sku_agg.head(50).copy()
        display_df.columns = ["国家", "品类", "SKU", "30天总量", "日均", "峰值日量"]
        display_df["30天总量"] = display_df["30天总量"].astype(int)
        display_df["日均"]    = display_df["日均"].round(1)
        display_df["峰值日量"] = display_df["峰值日量"].astype(int)
        st.dataframe(display_df, use_container_width=True, height=480, hide_index=True)

    # 异常预警
    if "anomaly_flag" in fc.columns:
        st.markdown("---")
        st.markdown("**⚠ 异常预警（spike/dip SKU）**")
        anomaly = fc[fc["anomaly_flag"] != "normal"].groupby(
            ["country", "category", "sku_id", "anomaly_flag"]
        ).agg(
            dates=("date", lambda x: ", ".join(x.dt.strftime("%m-%d").tolist()[:5])),
            max_forecast=(fc_col, "max"),
        ).reset_index()

        if anomaly.empty:
            st.success("✅ 预测期内无异常 SKU")
        else:
            spike_df = anomaly[anomaly["anomaly_flag"] == "spike"]
            dip_df   = anomaly[anomaly["anomaly_flag"] == "dip"]
            if not spike_df.empty:
                st.markdown("🔴 **需求暴增（spike）**")
                st.dataframe(spike_df.drop(columns=["anomaly_flag"]), use_container_width=True, hide_index=True)
            if not dip_df.empty:
                st.markdown("🔵 **需求骤降（dip）**")
                st.dataframe(dip_df.drop(columns=["anomaly_flag"]), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
