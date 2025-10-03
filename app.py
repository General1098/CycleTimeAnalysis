
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict

import pandas as pd
import streamlit as st
import altair as alt

# -------------------------------
# Optional integration hook
# -------------------------------
TIMEPIECE_AVAILABLE = False
try:
    import timepiece_integration  # must provide get_completed_items(team, start, end) -> DataFrame
    TIMEPIECE_AVAILABLE = hasattr(timepiece_integration, "get_completed_items")
except Exception:
    TIMEPIECE_AVAILABLE = False


# -------------------------------
# Utility Functions
# -------------------------------

def _to_month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz=ts.tz)

@st.cache_data(show_spinner=False)
def build_cycle_time_timeseries(
    df: pd.DataFrame,
    completed_col: str = "completed_date",
    ct_col: str = "cycle_time_days",
    tz: str = "Europe/London",
    rolling_days: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Build both monthly-bucketed and rolling-window (trailing N days) cycle-time series.
    Expects df with:
      - completed_col: datetime (can be naive or tz-aware)
      - ct_col: numeric (cycle time in days)
    Returns:
      - { 'monthly': DataFrame, 'rolling': DataFrame }
    """
    if df is None or df.empty:
        return {"monthly": pd.DataFrame(), "rolling": pd.DataFrame()}

    d = df.copy()

    # Ensure datetime and drop invalids
    d[completed_col] = pd.to_datetime(d[completed_col], utc=True, errors="coerce")
    d = d.dropna(subset=[completed_col, ct_col])
    # Localize to target tz so month boundaries match UK teams
    d[completed_col] = d[completed_col].dt.tz_convert(tz)

    # ---- Monthly ----
    d["month"] = d[completed_col].dt.to_period("M").dt.to_timestamp().dt.tz_localize(tz)
    monthly = (
        d.groupby("month", as_index=False)
         .agg(mean_ct=(ct_col, "mean"),
              median_ct=(ct_col, "median"),
              count=(ct_col, "size"))
         .sort_values("month")
         .reset_index(drop=True)
    )
    now_tz = pd.Timestamp.now(tz=tz)
    current_month_start = _to_month_start(now_tz)
    monthly["is_current_month"] = monthly["month"] == current_month_start

    # ---- Rolling (trailing window) ----
    start_date = d[completed_col].min().floor("D")
    end_date = d[completed_col].max().ceil("D")
    if pd.isna(start_date) or pd.isna(end_date):
        return {"monthly": monthly, "rolling": pd.DataFrame()}

    day_index = pd.date_range(start=start_date, end=end_date, freq="D", tz=tz)

    means, medians, counts = [], [], []
    for day in day_index:
        window_start = day - pd.Timedelta(days=rolling_days - 1)
        mask = (d[completed_col] >= window_start) & (d[completed_col] <= day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        window_vals = d.loc[mask, ct_col]
        c = int(window_vals.shape[0])
        counts.append(c)
        means.append(float(window_vals.mean()) if c else None)
        medians.append(float(window_vals.median()) if c else None)

    rolling = pd.DataFrame({
        "date": day_index,
        "mean_ct": means,
        "median_ct": medians,
        "count": counts,
    })

    return {"monthly": monthly, "rolling": rolling}


# -------------------------------
# Data Loading
# -------------------------------

@st.cache_data(show_spinner=True)
def load_completed_items(team: Optional[str], start: Optional[datetime], end: Optional[datetime]) -> pd.DataFrame:
    """Load completed items for a team and date range via integration hook (if present).
    Must return columns: completed_date (datetime), cycle_time_days (float), team (str).
    """
    if TIMEPIECE_AVAILABLE:
        try:
            df = timepiece_integration.get_completed_items(team=team, start=start, end=end)
            if not df.empty:
                required_cols = {"completed_date", "cycle_time_days"}
                missing = required_cols - set(df.columns)
                if missing:
                    st.warning(f"Missing expected columns from integration: {missing}. The chart may be incomplete.")
            return df
        except Exception as e:
            st.error(f"Error loading data from timepiece_integration: {e}")
            return pd.DataFrame(columns=["completed_date", "cycle_time_days", "team"]).head(0)
    else:
        return pd.DataFrame(columns=["completed_date", "cycle_time_days", "team"]).head(0)


# -------------------------------
# Chart Renderers
# -------------------------------

def render_cycle_time_trend(df_done: pd.DataFrame):
    st.subheader("Cycle Time Trend")

    with st.container():
        c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1.35])
        view = c1.selectbox("Time grouping", ["Monthly", "Rolling (30 days)", "Rolling (14 days)"], index=0)
        use_median = c2.checkbox("Show median", value=True)
        show_counts = c3.checkbox("Show item counts", value=True)
        include_current = c4.checkbox("Include current month", value=False,
                                      help="If off, current month is excluded from monthly chart to avoid early-month bias.")

        window = 30 if view.startswith("Rolling (30") else 14 if view.startswith("Rolling (14") else 30

    ts = build_cycle_time_timeseries(
        df_done,
        completed_col="completed_date",
        ct_col="cycle_time_days",
        tz="Europe/London",
        rolling_days=window,
    )

    try:
        ts30 = build_cycle_time_timeseries(df_done, rolling_days=30)
    except Exception:
        ts30 = {"rolling": pd.DataFrame()}

    last30 = ts30.get("rolling", pd.DataFrame())
    if last30 is not None and not last30.empty:
        last_row = last30.dropna(subset=["median_ct", "mean_ct", "count"]).tail(1)
        m1, m2, m3 = st.columns(3)
        med_val = last_row["median_ct"].iloc[0] if not last_row.empty else None
        mean_val = last_row["mean_ct"].iloc[0] if not last_row.empty else None
        cnt_val = int(last_row["count"].iloc[0]) if not last_row.empty else 0
        m1.metric("Median CT (30d)", f"{med_val:.1f} days" if med_val is not None else "—")
        m2.metric("Avg CT (30d)", f"{mean_val:.1f} days" if mean_val is not None else "—")
        m3.metric("Items (30d)", f"{cnt_val}")

    if view == "Monthly":
        monthly = ts["monthly"]
        if monthly.empty:
            st.info("No completed items to plot.")
            return

        if not include_current and "is_current_month" in monthly.columns:
            monthly = monthly[~monthly["is_current_month"]]

        y_field = "median_ct" if use_median else "mean_ct"
        y_title = "Median CT (days)" if use_median else "Average CT (days)"

        base = alt.Chart(monthly).encode(x=alt.X("month:T", title="Month"))

        line = base.mark_line(point=True).encode(
            y=alt.Y(f"{y_field}:Q", title=y_title),
            tooltip=[
                alt.Tooltip("month:T", title="Month"),
                alt.Tooltip(f"{y_field}:Q", title=y_title, format=".2f"),
                alt.Tooltip("count:Q", title="Items")
            ]
        )

        chart = line

        if include_current and "is_current_month" in ts["monthly"].columns:
            provisional = base.transform_filter(
                alt.datum.is_current_month == True
            ).mark_point(size=120, opacity=0.2).encode(
                y=f"{y_field}:Q",
                tooltip=[alt.Tooltip("month:T", title="Provisional Month")],
            )
            chart = (line + provisional)

        if show_counts:
            bars = base.mark_bar(opacity=0.2).encode(y=alt.Y("count:Q", title="Items (count)", axis=alt.Axis(titleColor="#666")))
            labels = base.mark_text(dy=-18).encode(text="count:Q")
            chart = (bars + line + labels)

        st.altair_chart(chart.properties(height=380), use_container_width=True)

        small = monthly["count"].iloc[-1] if not monthly.empty else 0
        if include_current and any(monthly.get("is_current_month", [])) and small < 5:
            st.caption("Current month CT is provisional due to low sample size.")

    else:
        rolling = ts["rolling"]
        if rolling.empty:
            st.info("No completed items to plot.")
            return

        y_field = "median_ct" if use_median else "mean_ct"
        y_title = f"{'Median' if use_median else 'Average'} CT (days) — trailing {window}d"

        base = alt.Chart(rolling).encode(x=alt.X("date:T", title="Date"))

        line = base.mark_line().encode(
            y=alt.Y(f"{y_field}:Q", title=y_title),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip(f"{y_field}:Q", title=y_title, format=".2f"),
                alt.Tooltip("count:Q", title=f"Items in last {window}d"),
            ],
        )
        chart = line

        if show_counts:
            area = base.mark_area(opacity=0.15).encode(
                y=alt.Y("count:Q", title=f"Items (last {window}d)"),
                tooltip=[alt.Tooltip("count:Q", title=f"Items (last {window}d)")],
            )
            chart = (area + line)

        st.altair_chart(chart.properties(height=380), use_container_width=True)


# -------------------------------
# Streamlit Page
# -------------------------------

def layout_sidebar():
    st.sidebar.header("Filters")

    team = st.sidebar.text_input("Team (optional)", value="")
    today = datetime.now()
    default_start = today - timedelta(days=180)
    start_date = st.sidebar.date_input("From", value=default_start.date())
    end_date = st.sidebar.date_input("To", value=today.date())

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    st.sidebar.caption("Tip: Use Rolling 30d view mid-month to avoid early-month bias.")
    return team.strip() or None, start_dt, end_dt

def main():
    st.set_page_config(page_title="Cycle Time Analysis", layout="wide")

    st.title("Cycle Time Analysis")

    team, start_dt, end_dt = layout_sidebar()
    df_done = load_completed_items(team=team, start=start_dt, end=end_dt)

    tab1, = st.tabs(["Cycle Time"])
    with tab1:
        render_cycle_time_trend(df_done)

    st.markdown("---")
    source_str = "Timepiece" if TIMEPIECE_AVAILABLE else "(no integration detected)"
    st.caption(f"Data source: {source_str}. Timezone: Europe/London.")

if __name__ == "__main__":
    main()
