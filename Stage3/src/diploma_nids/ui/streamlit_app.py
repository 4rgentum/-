"""Streamlit dashboard for the NIDS prototype.

Connects to the FastAPI inference service over HTTP and shows:
    - alerts table (recent N)
    - score time-series
    - severity histogram
    - basic drift indicator (placeholder; updated via /alerts/recent metadata)

Run:
    streamlit run src/diploma_nids/ui/streamlit_app.py
"""
from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("DIPLOMA_API_URL", "http://127.0.0.1:8000")
REFRESH_INTERVAL = float(os.environ.get("DIPLOMA_UI_REFRESH", "2.0"))

st.set_page_config(page_title="diploma-nids dashboard", layout="wide", page_icon=":shield:")
st.title("NIDS prototype dashboard")

with st.sidebar:
    st.header("Service")
    st.write(f"API: `{API_URL}`")
    st.write(f"Refresh: {REFRESH_INTERVAL:.1f} s")

    info: dict[str, Any] = {}
    try:
        info = requests.get(f"{API_URL}/info", timeout=2).json()
        st.success("connected")
    except Exception as exc:  # noqa: BLE001
        st.error(f"unreachable: {exc}")

    if info:
        st.write(f"loaded: **{info.get('loaded')}**")
        st.write(f"classical: {info.get('is_classical')}")
        st.write(f"threshold: {info.get('threshold')}")

st.subheader("Recent alerts")

placeholder_metrics = st.empty()
placeholder_table = st.empty()
placeholder_chart = st.empty()


def _fetch_alerts(n: int = 100) -> pd.DataFrame:
    try:
        resp = requests.get(f"{API_URL}/alerts/recent", params={"n": n}, timeout=2).json()
        alerts = resp.get("alerts", [])
        return pd.DataFrame(alerts)
    except Exception:
        return pd.DataFrame()


def _render(df: pd.DataFrame) -> None:
    if df.empty:
        placeholder_metrics.info("no alerts yet")
        placeholder_table.empty()
        placeholder_chart.empty()
        return

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["severity"] = df["severity"].astype("category")

    cols = placeholder_metrics.columns(4)
    cols[0].metric("alerts (recent)", len(df))
    cols[1].metric("critical", int((df["severity"] == "critical").sum()))
    cols[2].metric("high", int((df["severity"] == "high").sum()))
    cols[3].metric("avg score", f"{df['score'].mean():.3f}")

    placeholder_table.dataframe(
        df[["timestamp", "severity", "score", "decision", "attack_cat_pred", "alert_id"]].tail(50),
        use_container_width=True,
    )
    placeholder_chart.line_chart(df.set_index("timestamp")["score"])


# Polling loop (Streamlit re-runs the script on each refresh)
df = _fetch_alerts(100)
_render(df)

time.sleep(REFRESH_INTERVAL)
st.rerun()
