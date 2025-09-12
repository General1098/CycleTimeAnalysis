import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys, os
import datetime

# Monte Carlo Simulation
def monte_carlo_forecast(ct_data, num_simulations=10000, items=None, weeks=None):
    """
    Run a Monte Carlo simulation.
    - If items is provided: forecast how long it will take to finish X items.
    - If weeks is provided: forecast how many items can be delivered in N weeks.
    """
    ct_data = ct_data.dropna().values
    results = []

    if items:
        for _ in range(num_simulations):
            samples = np.random.choice(ct_data, size=items, replace=True)
            results.append(np.sum(samples))
        return np.percentile(results, [50, 85, 95])  # durations in days

    elif weeks:
        horizon_days = weeks * 7
        for _ in range(num_simulations):
            days_used = 0
            delivered = 0
            while days_used < horizon_days:
                days_used += np.random.choice(ct_data)
                if days_used <= horizon_days:
                    delivered += 1
            results.append(delivered)
        return np.percentile(results, [50, 85, 95])  # items delivered


# Ensure local imports work
sys.path.append(os.path.dirname(__file__))

from timepiece_integration import (
    fetch_status_durations,
    fetch_transition_dates,
    parse_status_rules,
    build_dataframe,
)

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


# ===================== SIDEBAR: SETTINGS =====================
with st.sidebar:
    st.title("Cycle Time Analysis (OBSS Timepiece API)")
    st.caption("Fetch Jira data via OBSS Timepiece Cloud")

    # Default API key hardcoded
    api_key = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjb20ub2Jzcy5wbHVnaW4udGltZS1pbi1zdGF0dXMiLCJzdWIiOiI2MzIxZDYyMWM3NjAxYzhlNGFiZjMxZjciLCJjbGllbnRLZXkiOiIwYjRjZjE5NC0yN2Q4LTM0MjItOGJmNi0wMWI1NzdiYzg4ZjUiLCJpc3MiOiJjb20ub2Jzcy5wbHVnaW4udGltZS1pbi1zdGF0dXMiLCJleHAiOjQxMDI0NDQ3NDAsImlhdCI6MTc1NzU5NTAxOH0.5shgcu7jHiQ4L5rTqFIHIhE-tiKnBsaab39LdyP_A4M"
    st.text(f"Using API Key: {api_key[:6]}...{api_key[-6:]}")

    filter_id = st.text_input("Saved Filter ID", value="10580")

    st.markdown("### Status Buckets")
    default_rules = """Blocked = On Hold (C7SM)
Development = In Development (C7SM), In Progress (C7O), In Development (C7T4)
Review = Review (C7SM)"""
    rules_text = st.text_area(
        "Rules (Bucket = Status1, Status2, ...)", value=default_rules, height=120
    )

    fetch_button = st.button("Fetch Data")

if not api_key or not filter_id:
    st.info("Enter Filter ID, then press **Fetch Data**.")
    st.stop()


# ===================== FETCH DATA FROM TIMEPIECE =====================
if fetch_button:
    with st.spinner("Fetching reports from Timepiece..."):
        try:
            duration_data = fetch_status_durations(api_key, filter_id)
            transition_data = fetch_transition_dates(api_key, filter_id)
            status_rules = parse_status_rules(rules_text)
            raw = build_dataframe(duration_data, transition_data, status_rules)

            # Assign teams automatically
            if not raw.empty and "Key" in raw.columns:
                raw["Team"] = raw["Key"].apply(assign_team)
            else:
                raw["Team"] = "Unknown"

            raw["_bucketer"] = pd.to_datetime(raw["End"], errors="coerce")

            # Cache results
            st.session_state["raw_data"] = raw
        except Exception as e:
            st.error(f"Failed to fetch from Timepiece: {e}")
            st.stop()

# Use cached data if available
if "raw_data" not in st.session_state:
    st.info("Fetch data first.")
    st.stop()

raw = st.session_state["raw_data"]

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
tabs = st.tabs(["Overview", "Cycle Time", "Slowest Items", "Forecasting", "Data"])


# ---------- OVERVIEW ----------
with tabs[0]:
    st.subheader(
        "Overview" + ("" if selected_team == "All Teams" else f" — {selected_team}")
    )
    col1, col2, col3 = st.columns(3)

    avg_ct = round(view_df["CT"].mean(), 2) if view_df["CT"].notna().any() else np.nan
    p85_ct = (
        round(view_df["CT"].quantile(0.85), 2) if view_df["CT"].notna().any() else np.nan
    )

    if not view_df.empty and view_df["_bucketer"].notna().any():
        items_this_month = (
            view_df["_bucketer"]
            .dropna()
            .dt.to_period("M")
            .value_counts()
            .sort_index()
            .iloc[-1]
        )
    else:
        items_this_month = 0

    col1.metric("Average CT (days)", "--" if np.isnan(avg_ct) else avg_ct)
    col2.metric("85th Percentile CT (days)", "--" if np.isnan(p85_ct) else p85_ct)
    col3.metric("Items this month", int(items_this_month))


# ---------- CYCLE TIME ----------
with tabs[1]:
    st.subheader("Cycle Time Trends")
    if "_bucketer" in view_df.columns and view_df["_bucketer"].notna().any():
        roll = (
            view_df.groupby(view_df["_bucketer"].dt.to_period("M"))
            .agg(
                avg=("CT", "mean"),
                p85=("CT", lambda s: s.quantile(0.85)),
                items=("CT", "count"),
            )
            .reset_index()
        )
        roll["_bucketer"] = roll["_bucketer"].dt.to_timestamp()

        base = alt.Chart(roll).encode(
            x=alt.X("_bucketer:T", title="Month", sort="ascending")
        )
        st.altair_chart(
            base.mark_line(point=True).encode(y=alt.Y("avg:Q", title="Average CT (days)")),
            use_container_width=True,
        )
        st.altair_chart(
            base.mark_line(point=True).encode(
                y=alt.Y("p85:Q", title="85th CT (days)")
            ),
            use_container_width=True,
        )
        st.altair_chart(
            base.mark_bar().encode(y=alt.Y("items:Q", title="Item Count")),
            use_container_width=True,
        )
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
                slow[["Key", "Team", "CT", "Start", "End"]]
                .sort_values("CT", ascending=False)
                .head(200),
                use_container_width=True,
            )
    else:
        st.info("No data available.")


# ---------- FORECASTING ----------
with tabs[3]:
    st.subheader("Monte Carlo Forecasting")

    if view_df.empty or view_df["CT"].dropna().empty:
        st.info("No CT data available for forecasting.")
    else:
        mode = st.radio(
            "Forecast mode",
            ["How many items in N weeks?", "When will X items be done?"],
            index=0,
        )

        start_date = st.date_input("Start date", datetime.date.today())

        if mode == "How many items in N weeks?":
            weeks = st.number_input("Weeks", min_value=1, max_value=52, value=4)
            sims = st.number_input(
                "Simulations", min_value=1000, max_value=50000, value=10000, step=1000
            )
            p50, p85, p95 = monte_carlo_forecast(
                view_df["CT"], num_simulations=sims, weeks=weeks
            )
            st.write(f"In {weeks} weeks (starting {start_date:%d %b %Y}):")
            st.write(f"- **50% likely**: {int(p50)} items")
            st.write(f"- **85% likely**: {int(p85)} items")
            st.write(f"- **95% likely**: {int(p95)} items")

        else:
            items = st.number_input(
                "Number of items", min_value=1, max_value=200, value=10
            )
            sims = st.number_input(
                "Simulations", min_value=1000, max_value=50000, value=10000, step=1000
            )
            p50, p85, p95 = monte_carlo_forecast(
                view_df["CT"], num_simulations=sims, items=items
            )
            st.write(f"To deliver {items} items (starting {start_date:%d %b %Y}):")
            st.write(
                f"- **50% likely**: {(start_date + datetime.timedelta(days=p50)):%d %b %Y}"
            )
            st.write(
                f"- **85% likely**: {(start_date + datetime.timedelta(days=p85)):%d %b %Y}"
            )
            st.write(
                f"- **95% likely**: {(start_date + datetime.timedelta(days=p95)):%d %b %Y}"
            )


# ---------- DATA ----------
with tabs[4]:
    st.subheader("Data preview")

    row_count = len(view_df)
    st.caption(f"Showing all {row_count} rows returned from Timepiece")

    st.dataframe(view_df, use_container_width=True)
