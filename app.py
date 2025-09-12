import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys, os

# Ensure local imports work
sys.path.append(os.path.dirname(__file__))

from timepiece_integration import fetch_status_durations, fetch_transition_dates, parse_status_rules, build_dataframe

st.set_page_config(page_title="Cycle Time Analysis", layout="wide")

# ---- Team detection from issue key ----
def assign_team(issue_key: str) -> str:
    if issue_key.startswith("C7SM"):
        return "Team 1"
    elif issue_key.startswith("C7O"):
        return "Team 2"
    elif issue_key.startswith("C7T4"):
        return "Team 4"
    return "Other"

# ===================== SIDEBAR =====================
with st.sidebar:
    st.title("Cycle Time Analysis (OBSS Timepiece API)")
    api_key = st.text_input("TISJWT Token", type="password")
    filter_id = st.text_input("Saved Filter ID", value="10580")

    st.markdown("### Status Buckets")
    default_rules = """Blocked = On Hold (C7SM)
Development = In Development (C7SM), In Progress (C7O), In Development (C7T4)
Review = Review (C7SM)"""
    rules_text = st.text_area("Rules (Bucket = Status1, Status2, ...)", value=default_rules, height=120)

    fetch_button = st.button("Fetch Data")

if not fetch_button or not api_key or not filter_id:
    st.info("Enter token, Filter ID, and press **Fetch Data**.")
    st.stop()

# ===================== FETCH DATA =====================
with st.spinner("Fetching reports from Timepiece..."):
    try:
        duration_data = fetch_status_durations(api_key, filter_id)
        transition_data = fetch_transition_dates(api_key, filter_id)
        status_rules = parse_status_rules(rules_text)
        raw = build_dataframe(duration_data, transition_data, status_rules)

        if not raw.empty and "Key" in raw.columns:
            raw["Team"] = raw["Key"].apply(assign_team)
        else:
            raw["Team"] = "Unknown"

        raw["_bucketer"] = pd.to_datetime(raw["End"], errors="coerce")

    except Exception as e:
        st.error(f"Failed to fetch from Timepiece: {e}")
        st.stop()

if raw.empty:
    st.warning("No data returned from Timepiece.")
    st.stop()

# ===================== MAIN PAGE =====================
st.title("Cycle Time Analysis")

teams = sorted(raw["Team"].unique().tolist())
selected_team = st.selectbox("Select Team", ["All Teams"] + teams, index=0)

if selected_team == "All Teams":
    view_df = raw.copy()
else:
    view_df = raw[raw["Team"] == selected_team].copy()

st.caption("Cycle Time = Blocked + Development + Review (if present)")

# ===================== TABS =====================
tabs = st.tabs(["Overview", "Cycle Time", "Slowest Items", "Sprint Analysis", "Data"])

# ---------- OVERVIEW ----------
with tabs[0]:
    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)

    avg_ct = round(view_df["CT"].mean(), 2) if view_df["CT"].notna().any() else np.nan
    p85_ct = round(view_df["CT"].quantile(0.85), 2) if view_df["CT"].notna().any() else np.nan
    items_this_month = (
        view_df["_bucketer"].dt.to_period("M").value_counts().sort_index().iloc[-1]
        if view_df["_bucketer"].notna().any()
        else 0
    )

    col1.metric("Average CT (days)", "--" if np.isnan(avg_ct) else avg_ct)
    col2.metric("85th Percentile CT (days)", "--" if np.isnan(p85_ct) else p85_ct)
    col3.metric("Items this month", int(items_this_month))

# ---------- CYCLE TIME ----------
with tabs[1]:
    st.subheader("Cycle Time Trends")
    if "_bucketer" in view_df.columns and view_df["_bucketer"].notna().any():
        roll = (
            view_df.groupby(view_df["_bucketer"].dt.to_period("M"))
            .agg(avg=("CT", "mean"), p85=("CT", lambda s: s.quantile(0.85)), items=("CT", "count"))
            .reset_index()
        )
        roll["_bucketer"] = roll["_bucketer"].dt.to_timestamp()

        base = alt.Chart(roll).encode(x=alt.X("_bucketer:T", title="Month"))
        st.altair_chart(base.mark_line(point=True).encode(y="avg:Q"), use_container_width=True)
        st.altair_chart(base.mark_line(point=True).encode(y="p85:Q"), use_container_width=True)
        st.altair_chart(base.mark_bar().encode(y="items:Q"), use_container_width=True)
    else:
        st.info("No valid transition dates found.")

# ---------- SLOWEST ITEMS ----------
with tabs[2]:
    st.subheader("Slowest Items")
    th = view_df["_bucketer"].dt.to_period("M")
    p85 = view_df.groupby(th)["CT"].quantile(0.85).rename("p85")
    merged = view_df.join(p85, on=th)
    slow = merged[merged["CT"] > merged["p85"]]

    if slow.empty:
        st.info("No items above 85th percentile.")
    else:
        st.dataframe(slow[["Key", "Team", "CT", "Start", "End"]].sort_values("CT", ascending=False), use_container_width=True)

# ---------- SPRINT ANALYSIS ----------
with tabs[3]:
    st.subheader("Sprint Analysis")

    if "Sprint" not in view_df.columns or view_df["Sprint"].isna().all():
        st.info("No sprint data available. Ensure customfield_10021 is included.")
    else:
        sprint_summary = (
            view_df.groupby(["Team", "Sprint", "IssueType"]).size().reset_index(name="Count")
        )
        sprint_summary["Total"] = sprint_summary.groupby(["Team", "Sprint"])["Count"].transform("sum")
        sprint_summary["Percent"] = (sprint_summary["Count"] / sprint_summary["Total"] * 100).round(1)

        # Order sprints by numeric suffix if possible
        def sprint_order(name):
            import re
            m = re.search(r"(\d+)$", str(name))
            return int(m.group(1)) if m else float("inf")
        sprint_summary["SprintOrder"] = sprint_summary["Sprint"].apply(sprint_order)

        st.dataframe(sprint_summary.sort_values(["Team", "SprintOrder", "IssueType"]), use_container_width=True)

        # Visualization
        chart = (
            alt.Chart(sprint_summary)
            .mark_bar()
            .encode(
                x=alt.X("Sprint:N", sort=alt.SortField("SprintOrder", order="ascending")),
                y=alt.Y("Percent:Q", stack="normalize"),
                color="IssueType:N",
                column="Team:N"
            )
        )
        st.altair_chart(chart, use_container_width=True)

        # Carried-over analysis
        st.subheader("Carried-Over Items")
        carried = (
            raw.groupby("Key")["Sprint"]
            .nunique()
            .reset_index(name="SprintCount")
            .merge(raw[["Key", "Team", "IssueType"]], on="Key", how="left")
            .drop_duplicates("Key")
        )
        carried = carried[carried["SprintCount"] > 1]

        if carried.empty:
            st.info("No carried-over items detected.")
        else:
            st.dataframe(carried, use_container_width=True)

# ---------- DATA ----------
with tabs[4]:
    st.subheader("Data Preview")
    st.dataframe(view_df, use_container_width=True)
