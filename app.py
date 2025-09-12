import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys, os

# Ensure local imports work
sys.path.append(os.path.dirname(__file__))

from timepiece_integration import fetch_status_durations, fetch_transition_dates, parse_status_rules, build_dataframe

st.set_page_config(page_title="Cycle Time Analysis", layout="wide")


# -------------------------------
# Helper: format CT nicely
# -------------------------------
def format_days(days: float) -> str:
    if pd.isna(days):
        return "-"
    total_minutes = int(days * 24 * 60)
    d, rem = divmod(total_minutes, 1440)
    h, m = divmod(rem, 60)
    parts = []
    if d > 0:
        parts.append(f"{d}d")
    if h > 0:
        parts.append(f"{h}h")
    if d == 0 and m > 0:  # only show minutes if < 1 day
        parts.append(f"{m}m")
    return " ".join(parts) if parts else "0d"


# ===================== SIDEBAR: SETTINGS =====================
with st.sidebar:
    st.title("Cycle Time Analysis (OBSS Timepiece API)")
    st.caption("Fetch Jira data via OBSS Timepiece Cloud")

    # Hardcoded API Key
    api_key = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjb20ub2Jzcy5wbHVnaW4udGltZS1pbi1zdGF0dXMiLCJzdWIiOiI2MzIxZDYyMWM3NjAxYzhlNGFiZjMxZjciLCJjbGllbnRLZXkiOiIwYjRjZjE5NC0yN2Q4LTM0MjItOGJmNi0wMWI1NzdiYzg4ZjUiLCJpc3MiOiJjb20ub2Jzcy5wbHVnaW4udGltZS1pbi1zdGF0dXMiLCJleHAiOjQxMDI0NDQ3NDAsImlhdCI6MTc1NzU5NTAxOH0.5shgcu7jHiQ4L5rTqFIHIhE-tiKnBsaab39LdyP_A4M"
    st.text(f"Using API Key: {api_key[:6]}...{api_key[-6:]}")

    filter_id = st.text_input("Saved Filter ID", value="10580")
    team = st.text_input("Team Name", value="Team 1")

    st.markdown("### Status Buckets")
    default_rules = """Blocked = On Hold (C7SM)
Development = In Development (C7SM), In Progress (C7O), In Development (C7T4)
Review = Review (C7SM)"""
    rules_text = st.text_area("Rules (Bucket = Status1, Status2, ...)", value=default_rules, height=120)

    fetch_button = st.button("Fetch Data")

if not fetch_button or not api_key or not filter_id:
    st.info("Enter Filter ID and press **Fetch Data**.")
    st.stop()


# ===================== FETCH DATA FROM TIMEPIECE =====================
with st.spinner("Fetching reports from Timepiece..."):
    try:
        duration_data = fetch_status_durations(api_key, filter_id)
        transition_data = fetch_transition_dates(api_key, filter_id)
    except Exception as e:
        st.error(f"Failed to fetch from Timepiece: {e}")
        st.stop()


# ===================== PROCESS DATA =====================
status_rules = parse_status_rules(rules_text)
raw = build_dataframe(duration_data, transition_data, status_rules)
raw["Team"] = team
raw["_bucketer"] = pd.to_datetime(raw["End"], errors="coerce")
raw["CT_pretty"] = raw["CT"].apply(format_days)  # add pretty CT

if raw.empty:
    st.warning("No data returned from Timepiece.")
    st.stop()


# ===================== MAIN PAGE =====================
st.title("Cycle Time Analysis")

teams = sorted(raw["Team"].unique().tolist())
selected_team = st.selectbox("Select Team", ["All Teams"] + teams, index=0)

# Filter the view
if selected_team == "All Teams":
    view_df = raw.copy()
else:
    view_df = raw[raw["Team"] == selected_team].copy()

st.caption("Cycle Time = Blocked + Development + Review (if present)")


# ===================== TABS =====================
tabs = st.tabs(["Overview", "Cycle Time", "Slowest Items", "Data"])


# ---------- OVERVIEW ----------
with tabs[0]:
    st.subheader("Overview" + ("" if selected_team == "All Teams" else f" — {selected_team}"))
    col1, col2, col3 = st.columns(3)

    avg_ct = round(view_df["CT"].mean(), 2) if view_df["CT"].notna().any() else np.nan
    p85_ct = round(view_df["CT"].quantile(0.85), 2) if view_df["CT"].notna().any() else np.nan
    items_this_month = (
        view_df["_bucketer"].dt.to_period("M").value_counts().sort_index().iloc[-1]
        if not view_df.empty
        else 0
    )

    col1.metric("Average CT (days)", "--" if np.isnan(avg_ct) else avg_ct)
    col2.metric("85th Percentile CT (days)", "--" if np.isnan(p85_ct) else p85_ct)
    col3.metric("Items this month", int(items_this_month))


# ---------- CYCLE TIME ----------
with tabs[1]:
    st.subheader("Cycle Time Trends")
    if "_bucketer" in view_df.columns and view_df["_bucketer"].notna().any():
        roll = view_df.groupby(view_df["_bucketer"].dt.to_period("M")).agg(
            avg=("CT", "mean"),
            p85=("CT", lambda s: s.quantile(0.85)),
            items=("CT", "count"),
        ).reset_index()
        roll["_bucketer"] = roll["_bucketer"].dt.to_timestamp()

        base = alt.Chart(roll).encode(x=alt.X("_bucketer:T", title="Month", sort="ascending"))
        st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("avg:Q", title="Average CT (days)")), use_container_width=True)
        st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("p85:Q", title="85th CT (days)")), use_container_width=True)
        st.altair_chart(base.mark_bar().encode(y=alt.Y("items:Q", title="Item Count")), use_container_width=True)
    else:
        st.info("No valid transition dates found for bucketing.")


# ---------- SLOWEST ITEMS ----------
with tabs[2]:
    st.subheader("Slowest Items")
    if not view_df.empty:
        th = view_df["_bucketer"].dt.to_period("M")
        p85 = view_df.groupby(th)["CT"].quantile(0.85).rename("p85")
        merged = view_df.join(p85, on=th)
        slow = merged[merged["CT"] > merged["p85"]].copy()

        if slow.empty:
            st.info("No items above the 85th percentile.")
        else:
            st.dataframe(
                slow[["Key", "Team", "CT_pretty", "Start", "End"]].sort_values("CT", ascending=False).head(200),
                use_container_width=True,
            )
    else:
        st.info("No data available.")


# ---------- DATA ----------
with tabs[3]:
    st.subheader("Data preview")
    st.dataframe(view_df[["Key", "Team", "CT", "CT_pretty", "Start", "End"]].head(1000), use_container_width=True)
