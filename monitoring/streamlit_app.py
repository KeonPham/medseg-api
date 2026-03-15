"""Streamlit monitoring dashboard for MedSeg API.

Displays prediction statistics, model performance metrics,
and data drift detection results.

Run:
    streamlit run monitoring/streamlit_app.py
"""

import os
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

DB_PATH = os.environ.get("MEDSEG_DB", "predictions.db")
REGISTRY_PATH = os.environ.get("MEDSEG_REGISTRY", "configs/model_registry.yaml")


@st.cache_resource
def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def load_predictions(hours: int) -> pd.DataFrame:
    """Load predictions from the last N hours."""
    conn = get_connection()
    since = (datetime.now() - timedelta(hours=hours)).isoformat()
    query = """
        SELECT id, request_id, timestamp, model_name, model_version,
               inference_time_ms, image_hash, confidence_score,
               lung_coverage_pct, symmetry_ratio
        FROM predictions
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
    """
    df = pd.read_sql_query(query, conn, params=(since,))
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_registry() -> dict:
    """Load model registry YAML."""
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f) or {}


# ── Page config ────────────────────────────────────────────

st.set_page_config(page_title="MedSeg Monitor", page_icon="🫁", layout="wide")
st.title("MedSeg API Monitoring Dashboard")

# ── Sidebar ────────────────────────────────────────────────

hours = st.sidebar.slider("Look-back (hours)", 1, 168, 24)
tab_overview, tab_performance, tab_quality, tab_registry = st.tabs(
    ["Overview", "Performance", "Quality", "Model Registry"]
)

df = load_predictions(hours)

# ── Overview tab ───────────────────────────────────────────

with tab_overview:
    if df.empty:
        st.info(f"No predictions in the last {hours} hours.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", len(df))
        col2.metric("Avg Latency (ms)", f"{df['inference_time_ms'].mean():.1f}")
        col3.metric("Avg Confidence", f"{df['confidence_score'].mean():.3f}")
        col4.metric("Unique Images", df["image_hash"].nunique())

        # 7-day count for context
        df_7d = load_predictions(168)
        st.caption(f"7-day total: {len(df_7d)} predictions")

        # Model distribution pie chart
        st.subheader("Model Distribution")
        model_counts = df["model_name"].value_counts().reset_index()
        model_counts.columns = ["model", "count"]
        fig_pie = px.pie(
            model_counts,
            values="count",
            names="model",
            hole=0.4,
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

        # Predictions over time
        st.subheader("Predictions Over Time")
        df_hourly = df.set_index("timestamp").resample("1h").size().reset_index(name="count")
        fig_timeline = px.bar(df_hourly, x="timestamp", y="count")
        fig_timeline.update_layout(xaxis_title="Time", yaxis_title="Predictions")
        st.plotly_chart(fig_timeline, use_container_width=True)


# ── Performance tab ───────────────────────────────────────

with tab_performance:
    if df.empty:
        st.info("No data available.")
    else:
        # Latency time series
        st.subheader("Inference Latency Over Time")
        fig_latency = px.scatter(
            df,
            x="timestamp",
            y="inference_time_ms",
            color="model_name",
            opacity=0.6,
        )
        fig_latency.update_layout(yaxis_title="Latency (ms)", xaxis_title="Time")
        st.plotly_chart(fig_latency, use_container_width=True)

        # Latency histogram
        st.subheader("Latency Distribution")
        fig_hist = px.histogram(
            df,
            x="inference_time_ms",
            color="model_name",
            nbins=50,
            barmode="overlay",
            opacity=0.7,
        )
        fig_hist.update_layout(xaxis_title="Latency (ms)", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Per-model latency stats
        st.subheader("Latency by Model")
        latency_stats = (
            df.groupby("model_name")["inference_time_ms"]
            .agg(["count", "mean", "median", "min", "max", "std"])
            .round(2)
        )
        st.dataframe(latency_stats, use_container_width=True)

        # P50 / P95 / P99
        st.subheader("Latency Percentiles")
        percentiles = (
            df.groupby("model_name")["inference_time_ms"]
            .quantile([0.5, 0.95, 0.99])
            .unstack()
            .round(2)
        )
        percentiles.columns = ["P50", "P95", "P99"]
        st.dataframe(percentiles, use_container_width=True)


# ── Quality tab ────────────────────────────────────────────

with tab_quality:
    if df.empty:
        st.info("No data available.")
    else:
        # Confidence distribution
        st.subheader("Confidence Score Distribution")
        fig_conf = px.histogram(
            df,
            x="confidence_score",
            nbins=40,
            color="model_name",
            barmode="overlay",
            opacity=0.7,
        )
        fig_conf.update_layout(xaxis_title="Confidence Score", yaxis_title="Count")
        st.plotly_chart(fig_conf, use_container_width=True)

        # Lung coverage trend
        st.subheader("Lung Coverage Over Time")
        fig_cov = px.scatter(
            df,
            x="timestamp",
            y="lung_coverage_pct",
            color="model_name",
            opacity=0.6,
            trendline="lowess",
        )
        fig_cov.update_layout(yaxis_title="Lung Coverage (%)", xaxis_title="Time")
        st.plotly_chart(fig_cov, use_container_width=True)

        # Symmetry ratio distribution
        st.subheader("Symmetry Ratio Distribution")
        fig_sym = px.histogram(
            df,
            x="symmetry_ratio",
            nbins=40,
            color="model_name",
            barmode="overlay",
            opacity=0.7,
        )
        fig_sym.update_layout(xaxis_title="Symmetry Ratio", yaxis_title="Count")
        st.plotly_chart(fig_sym, use_container_width=True)

        # Low-confidence alerts
        low_conf = df[df["confidence_score"] < 0.5]
        if not low_conf.empty:
            st.warning(f"{len(low_conf)} predictions with confidence < 0.5")
            st.dataframe(
                low_conf[["timestamp", "model_name", "confidence_score", "lung_coverage_pct"]].head(
                    20
                ),
                use_container_width=True,
            )


# ── Model Registry tab ────────────────────────────────────

with tab_registry:
    registry = load_registry()
    models_cfg = registry.get("models", {})
    default_model = registry.get("default_model", "N/A")

    st.subheader(f"Registered Models (default: {default_model})")

    if not models_cfg:
        st.info("No models in registry.")
    else:
        rows = []
        for model_name, model_def in models_cfg.items():
            arch = model_def.get("architecture", "unknown")
            for version, vdef in model_def.get("versions", {}).items():
                metrics = vdef.get("metrics", {})
                rows.append(
                    {
                        "Model": model_name,
                        "Version": version,
                        "Architecture": arch,
                        "Path": vdef.get("path", ""),
                        "Dice": metrics.get("dice", ""),
                        "IoU": metrics.get("iou", ""),
                        "HD95": metrics.get("hd95", ""),
                    }
                )

        registry_df = pd.DataFrame(rows)
        st.dataframe(registry_df, use_container_width=True)

        # Dice comparison chart
        if any(r["Dice"] for r in rows):
            valid = [r for r in rows if r["Dice"]]
            fig_dice = go.Figure()
            for r in valid:
                fig_dice.add_trace(
                    go.Bar(
                        x=[f"{r['Model']} ({r['Version']})"],
                        y=[r["Dice"]],
                        name=r["Model"],
                    )
                )
            fig_dice.update_layout(
                title="Dice Coefficient by Model Version",
                yaxis_title="Dice",
                showlegend=False,
            )
            st.plotly_chart(fig_dice, use_container_width=True)
