import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from timepiece_integration import run_timepiece_report, parse_status_rules, build_dataframe

st.set_page_config(page_title="Cycle Time Analysis", layout="wide")

# ===================== SIDEBAR: SETTINGS =====================
with st.sidebar:
    st.title("Cycle Time Analysis (Timepiece API)")
    st.caption("Fetch Jira data via Timepiece")

    api_key = st.text_input("API Key", type="password")
    jql_query = st.text_area("JQL Query", value="project = ABC AND statusCategory != Done")

    team = st.text_input("Team Name", value="Team 1")

    st.markdown("### Status Buckets")
    default_rules = """Development = In Dev, Implementation, Coding
Review = Code Review, QA, Test
On Hold = Blocked, Waiting, Paused"""
    rules_text = st.text_area("Rules (Bucket = Status1, Status2, ...)", value=default_rules, height=120)

    fetch_button = st.button("Fetch Data")

if not fetch_button or not api_key or not jql_query:
    st.info("Enter API key, JQL query, and press **Fetch Data**.")
    st.stop()

# ===================== FETCH DATA FROM TIMEPIECE =====================
with st.spinner("Fetching reports from Timepiece..."):
    try:
        duration_data = run_timepiece_report(api_key, "duration-by-status", jql_query)
        transition_data = run_timepiece_report(api_key, "transition-dates", jql_query)

    except Exception as e:
        st.error(f"Failed to fetch from Timepiece: {e}")
        st.stop()

# ===================== PROCESS DATA =====================
status_rules = parse_status_rules(rules_text)
raw = build_dataframe(duration_data, transition_data, status_rules)
raw["Team"] = team
raw["_bucketer"] = pd.to_datetime(raw["End"], errors="coerce")

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

st.caption("Cycle Time = sum of durations in On Hold + Development + Review")

# ===================== TABS =====================
tabs = st.tabs(["Overview", "Cycle Time", "Slowest Items", "Data"])

# ---------- OVERVIEW ----------
with tabs[0]:
    st.subheader("Overview" + ("" if selected_team == "All Teams" else f" — {selected_team}"))
    col1, col2, col3 = st.columns(3)

    avg_ct = round(view_df["CT"].mean(), 2) if view_df["CT"].notna().any() else np.nan
    p85_ct = round(view_df["CT"].quantile(0.85), 2) if view_df["CT"].notna().any() else np.nan
    items_this_month = view_df["_bucketer"].dt.to_period("M").value_counts().sort_index().iloc[-1] if not view_df.empty else 0

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
            items=("CT", "count")
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
            st.dataframe(slow[["Key", "Team", "CT", "Start", "End"]].sort_values("CT", ascending=False).head(200), use_container_width=True)
    else:
        st.info("No data available.")

# ---------- DATA ----------
with tabs[3]:
    st.subheader("Data preview")
    st.dataframe(view_df.head(1000), use_container_width=True)
