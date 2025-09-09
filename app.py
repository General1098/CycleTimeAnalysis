from __future__ import annotations

import io
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dateutil import parser as dtparser

from timepiece_client import TimepieceClient

# ----------------------------------
# Streamlit config
# ----------------------------------
st.set_page_config(page_title="Cycle Time Tracker", layout="wide")

DEFAULT_WORKDAY_HOURS = 8


# ----------------------------------
# Utility functions
# ----------------------------------
def parse_duration_text(text: str) -> Optional[timedelta]:
    """Parse strings like '0d 2h 36m 50s'; tolerate missing parts."""
    if not isinstance(text, str):
        return None
    pattern = r"\s*(?:(?P<d>\d+)\s*d)?\s*(?:(?P<h>\d+)\s*h)?\s*(?:(?P<m>\d+)\s*m)?\s*(?:(?P<s>\d+)\s*s)?\s*"
    m = re.fullmatch(pattern, text.strip())
    if not m:
        return None
    d = int(m.group("d") or 0)
    h = int(m.group("h") or 0)
    m_ = int(m.group("m") or 0)
    s = int(m.group("s") or 0)
    return timedelta(days=d, hours=h, minutes=m_, seconds=s)


def to_datetime_safe(x) -> Optional[datetime]:
    if pd.isna(x):
        return None
    if isinstance(x, datetime):
        return x
    try:
        return dtparser.parse(str(x))
    except Exception:
        return None


def ceil_timedelta(td: timedelta, unit: str) -> timedelta:
    seconds = td.total_seconds()
    if unit == "hours":
        hours = math.ceil(seconds / 3600.0)
        return timedelta(hours=hours)
    elif unit == "days":
        days = math.ceil(seconds / 86400.0)
        return timedelta(days=days)
    return td


def business_hours_diff(start: Optional[datetime], end: Optional[datetime], workday_hours: int = DEFAULT_WORKDAY_HOURS) -> timedelta:
    """Approximate business-time diff (Mon–Fri, no holidays)."""
    if start is None or end is None:
        return timedelta(0)
    if end < start:
        start, end = end, start

    start_date = start.date()
    end_date = end.date()

    # Excludes end_date, add one if the end day is a business day
    bdays = np.busday_count(start_date, end_date)
    if np.is_busday(end_date):
        bdays += 1

    hours = int(bdays) * int(workday_hours)
    return timedelta(hours=hours)


@dataclass
class CycleOptions:
    rounding: str  # 'days' | 'hours' | 'none'
    use_business_time: bool
    workday_hours: int


def compute_cycle_time(start: Optional[datetime], end: Optional[datetime], duration_text: Optional[str], opts: CycleOptions) -> Optional[timedelta]:
    td: Optional[timedelta] = None
    if start and end:
        td = business_hours_diff(start, end, opts.workday_hours) if opts.use_business_time else (end - start)
    elif duration_text:
        td = parse_duration_text(duration_text)

    if td is None:
        return None

    if opts.rounding == "hours":
        td = ceil_timedelta(td, "hours")
    elif opts.rounding == "days":
        td = ceil_timedelta(td, "days")
    return td


def td_to_hours(td: Optional[timedelta]) -> Optional[float]:
    if td is None:
        return None
    return td.total_seconds() / 3600.0


# ----------------------------------
# UI helpers
# ----------------------------------
def sidebar_options() -> CycleOptions:
    st.sidebar.header("Options")
    rounding = st.sidebar.selectbox("Rounding", ["days", "hours", "none"], index=0, help="Ceil rounding for computed durations")
    use_business = st.sidebar.checkbox("Use business time (Mon–Fri only)", value=True)
    workday_hours = st.sidebar.number_input("Workday hours", min_value=1, max_value=24, value=DEFAULT_WORKDAY_HOURS, step=1)
    return CycleOptions(rounding=rounding, use_business_time=use_business, workday_hours=workday_hours)


def show_metrics(df: pd.DataFrame, hours_col: str):
    vals = df[hours_col].dropna().values
    if len(vals) == 0:
        st.info("No cycle time values to summarise.")
        return
    p50 = float(np.percentile(vals, 50))
    p85 = float(np.percentile(vals, 85))
    p95 = float(np.percentile(vals, 95))
    avg = float(np.mean(vals))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median (P50)", f"{p50:.1f} h")
    c2.metric("P85", f"{p85:.1f} h")
    c3.metric("P95", f"{p95:.1f} h")
    c4.metric("Average", f"{avg:.1f} h")


def charts(df: pd.DataFrame, hours_col: str, id_col: str, start_col: Optional[str] = None):
    # Histogram
    hist = alt.Chart(df.dropna(subset=[hours_col])).mark_bar().encode(
        x=alt.X(f"{hours_col}:Q", bin=alt.Bin(maxbins=30), title="Cycle Time (hours)"),
        y=alt.Y("count()", title="Count"),
        tooltip=[alt.Tooltip(hours_col, title="Hours", format=".1f")]
    ).properties(height=300)
    st.altair_chart(hist, use_container_width=True)

    # Scatter by start date (if available)
    if start_col and start_col in df.columns:
        try:
            sdf = df.dropna(subset=[hours_col, start_col]).copy()
            sdf[start_col] = pd.to_datetime(sdf[start_col], errors="coerce")
            scatter = alt.Chart(sdf).mark_circle(size=60).encode(
                x=alt.X(f"{start_col}:T", title="Start Date"),
                y=alt.Y(f"{hours_col}:Q", title="Cycle Time (hours)"),
                tooltip=[id_col, alt.Tooltip(hours_col, format=".1f"), start_col]
            ).properties(height=320)
            st.altair_chart(scatter, use_container_width=True)
        except Exception:
            pass


# ----------------------------------
# Upload section
# ----------------------------------
def upload_section(opts: CycleOptions):
    st.subheader("Upload CSV / Excel")
    st.caption("Provide columns named **Start**, **End**, and/or **DurationText**. An **ID** column is recommended.")
    file = st.file_uploader("Upload file", type=["csv", "xlsx"])
    if not file:
        return

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    id_col = "ID" if "ID" in df.columns else df.columns[0]
    start_col = "Start" if "Start" in df.columns else None
    end_col = "End" if "End" in df.columns else None
    dur_col = "DurationText" if "DurationText" in df.columns else None

    def row_td(r):
        start = to_datetime_safe(r[start_col]) if start_col and start_col in r else None
        end = to_datetime_safe(r[end_col]) if end_col and end_col in r else None
        dur = r[dur_col] if dur_col and dur_col in r else None
        return compute_cycle_time(start, end, dur, opts)

    df["CycleTime_td"] = df.apply(row_td, axis=1)
    df["CycleTime_hours"] = df["CycleTime_td"].apply(td_to_hours)

    st.dataframe(df.head(50))
    show_metrics(df, "CycleTime_hours")
    charts(df, "CycleTime_hours", id_col, start_col=start_col)

    st.download_button("Download processed CSV", df.to_csv(index=False).encode("utf-8"), "processed_cycle_time.csv", "text/csv")


# ----------------------------------
# Timepiece section
# ----------------------------------
def timepiece_section(opts: CycleOptions):
    st.subheader("Timepiece (Time in Status)")
    st.caption("Fetch status durations directly from Timepiece using a TISJWT token.")

    default_host = st.secrets.get("timepiece", {}).get("api_host", "https://tis.obss.io")
    default_token = st.secrets.get("timepiece", {}).get("api_token", "")

    colA, colB = st.columns(2)
    api_host = colA.text_input("API Host", value=default_host, help="Timepiece host (usually https://tis.obss.io)")
    token = colB.text_input("TISJWT Token", value=default_token, type="password")

    safe_mode = st.checkbox("Safe Mode (smaller pages, minimal columns)", value=True)
    st.markdown("**Aggregation Parameters (JSON)**")

    minimal_params = {
        "page": 1,
        "pageSize": 50 if safe_mode else 100,
        "columns": ["Issue Key", "Time in Status (hours)"],
        # One of the following:
        # "jql": "project = ABC AND updated >= -14d",
        # "filterId": 12345,
    }
    params_text = st.text_area("Params JSON", value=json.dumps(minimal_params, indent=2), height=220)
    c1, c2, c3 = st.columns(3)
    fetch_btn = c1.button("Fetch (CSV)")
    json_btn = c2.button("Fetch (JSON)")
    test_btn = c3.button("Test Connection")

    if not token:
        st.info("Enter your TISJWT token to enable fetching.")
        return

    client = TimepieceClient(api_host=api_host, token=token)

    if test_btn:
        try:
            test_params = {"page": 1, "pageSize": 1, "columns": ["Issue Key"], "jql": "ORDER BY updated DESC"}
            _ = client.aggregation_json(test_params)
            st.success("Connection OK ✅")
        except Exception as ex:
            st.error(f"Test failed: {ex}")

    def handle_df(df: pd.DataFrame):
        if df.empty:
            st.warning("No rows returned from Timepiece.")
            return
        st.dataframe(df.head(50))

        numeric_cols = [c for c in df.columns if df[c].dtype.kind in "iuf"]
        st.markdown("### Map durations to Cycle Time")
        dur_cols = st.multiselect("Select numeric duration columns to sum (e.g., specific statuses)", options=numeric_cols, default=[c for c in numeric_cols if "Time in Status" in c][:1])
        units = st.selectbox("Units for selected columns", ["hours", "days"], index=0)

        if dur_cols:
            df["_CycleTimeSum"] = df[dur_cols].sum(axis=1).astype(float)
            df["CycleTime_hours"] = df["_CycleTimeSum"] * (24.0 if units == "days" else 1.0)

            # Rounding
            if opts.rounding in ("hours", "days"):
                mult = 1.0 if opts.rounding == "hours" else 24.0
                df["CycleTime_hours"] = np.ceil(df["CycleTime_hours"] / mult) * mult

            show_metrics(df, "CycleTime_hours")
            id_col = "Issue Key" if "Issue Key" in df.columns else df.columns[0]
            # Try a start-like column
            start_like = [c for c in df.columns if any(k in c.lower() for k in ["start", "created", "begin"])]
            start_col = start_like[0] if start_like else None
            charts(df, "CycleTime_hours", id_col, start_col=start_col)

            st.download_button("Download Timepiece + CycleTime CSV", df.to_csv(index=False).encode("utf-8"), "timepiece_cycle_time.csv", "text/csv")

    if fetch_btn:
        try:
            params = json.loads(params_text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return
        try:
            csv_bytes = client.aggregation_csv(params)
            df = pd.read_csv(io.BytesIO(csv_bytes))
            handle_df(df)
        except Exception as ex:
            st.error(f"CSV aggregation failed: {ex}")

    if json_btn:
        try:
            params = json.loads(params_text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return
        try:
            data = client.aggregation_json(params)
            # Convert typical JSON shape to DataFrame
            if isinstance(data, dict) and "rows" in data and "columns" in data:
                cols = [c.get("name") if isinstance(c, dict) else str(c) for c in data["columns"]]
                rows = data["rows"]
                df = pd.DataFrame(rows, columns=cols)
            else:
                df = pd.json_normalize(data)
            handle_df(df)
        except Exception as ex:
            st.error(f"JSON aggregation failed: {ex}")


# ----------------------------------
# Main
# ----------------------------------
def main():
    st.title("Cycle Time Tracker")
    st.caption("Upload CSV/Excel OR fetch directly from Timepiece (Time in Status).")

    opts = sidebar_options()
    tab1, tab2 = st.tabs(["Upload File", "Timepiece"])
    with tab1:
        upload_section(opts)
    with tab2:
        timepiece_section(opts)


if __name__ == "__main__":
    main()
