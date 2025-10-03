import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys, os
import datetime

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


# ---- Monte Carlo (throughput based) ----
def monte_carlo_forecast_throughput(completions_per_week, num_simulations=10000, items=None, weeks=None):
    results = []

    if items:  # Forecast completion time for X items
        for _ in range(num_simulations):
            total_done = 0
            week_count = 0
            while total_done < items:
                total_done += np.random.choice(completions_per_week)
                week_count += 1
            results.append(week_count * 7)  # convert to days
        return np.percentile(results, [50, 85, 95])

    elif weeks:  # Forecast number of items completed in N weeks
        for _ in range(num_simulations):
            total_done = 0
            for _ in range(weeks):
                total_done += np.random.choice(completions_per_week)
            results.append(total_done)
        return np.percentile(results, [50, 85, 95])


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
tabs = st.tabs(["Cycle Time", "Slowest Items", "Forecasting", "Data", "Sprint Analysis"])


# ---------- OVERVIEW ----------

with tabs[0]:
    
    # ---- Merged Overview & Insights (view-only) ----
    st.subheader("Cycle Time — Overview & Insights" + ("" if selected_team == "All Teams" else f" — {selected_team}"))
    col1, col2, col3, col4, col5 = st.columns(5)

    avg_ct = round(view_df["CT"].mean(), 2) if "CT" in view_df.columns and view_df["CT"].notna().any() else np.nan
    p85_ct = round(view_df["CT"].quantile(0.85), 2) if "CT" in view_df.columns and view_df["CT"].notna().any() else np.nan

    
with tabs[0]:
    st.subheader("Cycle Time Trends")

    # === Enhanced CT Trends (Monthly vs Rolling, Metric selector incl. P85, counts, provisional month) ===
    import pandas as _pd
    import altair as _alt

    def _build_ct_series_from_view(df: _pd.DataFrame, tz: str = "Europe/London", rolling_days: int = 30, min_n_for_pct: int = 20):
        if df is None or df.empty or "End" not in df.columns or "CT" not in df.columns:
            return {"monthly": _pd.DataFrame(), "rolling": _pd.DataFrame()}
        d = df[["End", "CT"]].dropna().copy()
        d.rename(columns={"End": "completed_date", "CT": "cycle_time_days"}, inplace=True)
        d["completed_date"] = _pd.to_datetime(d["completed_date"], utc=True, errors="coerce")
        d = d.dropna(subset=["completed_date", "cycle_time_days"])
        d["completed_date"] = d["completed_date"].dt.tz_convert(tz)

        # Monthly
        d["month"] = d["completed_date"].dt.to_period("M").dt.to_timestamp().dt.tz_localize(tz)
        monthly = (
            d.groupby("month", as_index=False)
             .agg(mean_ct=("cycle_time_days", "mean"),
                  median_ct=("cycle_time_days", "median"),
                  p85_ct=("cycle_time_days", lambda x: x.quantile(0.85) if len(x) >= min_n_for_pct else None),
                  count=("cycle_time_days", "size"))
             .sort_values("month")
             .reset_index(drop=True)
        )
        now_tz = _pd.Timestamp.now(tz=tz)
        current_month_start = _pd.Timestamp(year=now_tz.year, month=now_tz.month, day=1, tz=now_tz.tz)
        monthly["is_current_month"] = monthly["month"] == current_month_start

        # Rolling trailing window
        if d.empty:
            return {"monthly": monthly, "rolling": _pd.DataFrame()}
        start_date = d["completed_date"].min().floor("D")
        end_date = d["completed_date"].max().ceil("D")
        day_index = _pd.date_range(start=start_date, end=end_date, freq="D", tz=tz)

        means, medians, p85s, counts = [], [], [], []
        for day in day_index:
            window_start = day - _pd.Timedelta(days=rolling_days - 1)
            mask = (d["completed_date"] >= window_start) & (d["completed_date"] <= day + _pd.Timedelta(days=1) - _pd.Timedelta(seconds=1))
            vals = d.loc[mask, "cycle_time_days"]
            c = int(vals.shape[0]); counts.append(c)
            means.append(float(vals.mean()) if c else None)
            medians.append(float(vals.median()) if c else None)
            p85s.append(float(vals.quantile(0.85)) if c >= min_n_for_pct else None)

        rolling = _pd.DataFrame({"date": day_index, "mean_ct": means, "median_ct": medians, "p85_ct": p85s, "count": counts})
        return {"monthly": monthly, "rolling": rolling}

    ec1, ec2, ec3, ec4 = st.columns([1.3, 1.2, 1, 1.35])
    eview = ec1.selectbox("Time grouping", ["Monthly", "Rolling (30 days)", "Rolling (14 days)"], index=0, key="ctenh_view")
    metric = ec2.selectbox("Metric", ["Median (P50)", "Average (Mean)", "P85"], index=0, key="ctenh_metric")
    eshow_counts = ec3.checkbox("Show item counts", value=True, key="ctenh_counts")
    einclude_current = ec4.checkbox("Include current month", value=False, key="ctenh_current",
                                    help="If off, current month is excluded from monthly chart to avoid early-month bias.")

    ewindow = 30 if eview.startswith("Rolling (30") else 14 if eview.startswith("Rolling (14") else 30
    ets = _build_ct_series_from_view(view_df, rolling_days=ewindow)
    y_field_map = {"Median (P50)": "median_ct", "Average (Mean)": "mean_ct", "P85": "p85_ct"}
    y_field = y_field_map[metric]
    y_title = f"{metric} CT (days)"

    # KPI tiles from rolling 30d
    ets30 = _build_ct_series_from_view(view_df, rolling_days=30)
    last30 = ets30.get("rolling", _pd.DataFrame())
    if last30 is not None and not last30.empty:
        last_row = last30.dropna(subset=["median_ct", "mean_ct", "count"]).tail(1)
        m1, m2, m3 = st.columns(3)
        med_val = last_row["median_ct"].iloc[0] if not last_row.empty else None
        mean_val = last_row["mean_ct"].iloc[0] if not last_row.empty else None
        cnt_val = int(last30["count"].iloc[-1]) if not last30.empty else 0
        m1.metric("Median CT (30d)", f"{med_val:.1f} days" if med_val is not None else "—")
        m2.metric("Avg CT (30d)", f"{mean_val:.1f} days" if mean_val is not None else "—")
        m3.metric("Items (30d)", f"{cnt_val}")

    if eview == "Monthly":
        monthly = ets["monthly"]
        if monthly.empty:
            st.info("No completed items to plot.")
        else:
            if not einclude_current and "is_current_month" in monthly.columns:
                monthly = monthly[~monthly["is_current_month"]]

            base = _alt.Chart(monthly).encode(x=_alt.X("month:T", title="Month"))
            line = base.mark_line(point=True).encode(
                y=_alt.Y(f"{y_field}:Q", title=y_title),
                tooltip=[
                    _alt.Tooltip("month:T", title="Month"),
                    _alt.Tooltip(f"{y_field}:Q", title=y_title, format=".2f"),
                    _alt.Tooltip("count:Q", title="Items"),
                ],
            )
            chart = line
            if einclude_current and "is_current_month" in ets["monthly"].columns:
                provisional = base.transform_filter(_alt.datum.is_current_month == True).mark_point(size=120, opacity=0.2).encode(
                    y=f"{y_field}:Q",
                    tooltip=[_alt.Tooltip("month:T", title="Provisional Month")],
                )
                chart = (line + provisional)

            if eshow_counts:
                bars = base.mark_bar(opacity=0.2).encode(y=_alt.Y("count:Q", title="Items (count)", axis=_alt.Axis(titleColor="#666")))
                labels = base.mark_text(dy=-18).encode(text="count:Q")
                chart = (bars + line + labels)

            st.altair_chart(chart.properties(height=380), use_container_width=True)
    else:
        rolling = ets["rolling"]
        if rolling.empty:
            st.info("No completed items to plot.")
        else:
            base = _alt.Chart(rolling).encode(x=_alt.X("date:T", title="Date"))
            line = base.mark_line().encode(
                y=_alt.Y(f"{y_field}:Q", title=f"{y_title} — trailing {ewindow}d"),
                tooltip=[
                    _alt.Tooltip("date:T", title="Date"),
                    _alt.Tooltip(f"{y_field}:Q", title=y_title, format=".2f"),
                    _alt.Tooltip("count:Q", title=f"Items in last {ewindow}d"),
                ],
            )
            chart = line
            if eshow_counts:
                area = base.mark_area(opacity=0.15).encode(
                    y=_alt.Y("count:Q", title=f"Items (last {ewindow}d)"),
                    tooltip=[_alt.Tooltip("count:Q", title=f"Items (last {ewindow}d)")],
                )
                chart = (area + line)
            st.altair_chart(chart.properties(height=380), use_container_width=True)



# ---------- SLOWEST ITEMS ----------
with tabs[0]:
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
with tabs[0]:
    st.subheader("Monte Carlo Forecasting (Throughput Based)")

    if view_df.empty or view_df["_bucketer"].dropna().empty:
        st.info("No completion data available for forecasting.")
    else:
        # Build throughput history (items per week)
        throughput = (
            view_df["_bucketer"]
            .dropna()
            .dt.to_period("W")
            .value_counts()
            .values
        )
        if len(throughput) == 0:
            st.info("Not enough throughput data.")
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
                results = []
                for _ in range(sims):
                    total_done = 0
                    for _ in range(weeks):
                        total_done += np.random.choice(throughput)
                    results.append(total_done)

                p50, p85, p95 = np.percentile(results, [50, 85, 95])

                st.write(f"In {weeks} weeks (starting {start_date:%d %b %Y}):")
                st.write(f"- **50% likely**: {int(p50)} items")
                st.write(f"- **85% likely**: {int(p85)} items")
                st.write(f"- **95% likely**: {int(p95)} items")

                # Histogram
                chart_data = pd.DataFrame({"Items Delivered": results})
                hist = (
                    alt.Chart(chart_data)
                    .mark_bar()
                    .encode(
                        alt.X("Items Delivered:Q", bin=alt.Bin(maxbins=30)),
                        y="count()",
                    )
                )
                st.empty()  # legacy chart removed; st.altair_chart(hist, use_container_width=True)

            else:
                items = st.number_input(
                    "Number of items", min_value=1, max_value=200, value=10
                )
                sims = st.number_input(
                    "Simulations", min_value=1000, max_value=50000, value=10000, step=1000
                )
                results = []
                for _ in range(sims):
                    total_done = 0
                    week_count = 0
                    while total_done < items:
                        total_done += np.random.choice(throughput)
                        week_count += 1
                    results.append(week_count * 7)  # days

                p50, p85, p95 = np.percentile(results, [50, 85, 95])

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

                # Histogram
                chart_data = pd.DataFrame({"Days to Complete": results})
                hist = (
                    alt.Chart(chart_data)
                    .mark_bar()
                    .encode(
                        alt.X("Days to Complete:Q", bin=alt.Bin(maxbins=30)),
                        y="count()",
                    )
                )
                st.empty()  # legacy chart removed; st.altair_chart(hist, use_container_width=True)


# ---------- DATA ----------
with tabs[0]:
    st.subheader("Data preview")

    row_count = len(view_df)
    st.caption(f"Showing all {row_count} rows returned from Timepiece")

    st.dataframe(view_df, use_container_width=True)


# ---------- SPRINT ANALYSIS ----------
try:
    sprint_tab = tabs[-1]
except Exception:
    sprint_tab = None

if sprint_tab is not None:
    with sprint_tab:
        st.subheader("Sprint Analysis")

        df_src = view_df if 'view_df' in globals() else (df if 'df' in globals() else None)
        if df_src is None or df_src.empty or "Sprint" not in df_src.columns or "IssueType" not in df_src.columns:
            st.info("No sprint/issue type data available.")
        else:
            expanded = df_src.copy()
            expanded["Sprint"] = expanded["Sprint"].fillna("")
            expanded = expanded.assign(Sprint=expanded["Sprint"].astype(str).str.split(","))
            expanded = expanded.explode("Sprint")
            expanded["Sprint"] = expanded["Sprint"].astype(str).str.strip()
            expanded = expanded[expanded["Sprint"] != ""]

            if expanded.empty:
                st.info("No sprint data after expansion.")
            else:
                end_col = "End" if "End" in expanded.columns else ("Resolved" if "Resolved" in expanded.columns else None)
                expanded["_enddate"] = pd.to_datetime(expanded[end_col], errors="coerce") if end_col else pd.NaT

                sprint_order = (
                    expanded.groupby("Sprint")["_enddate"]
                    .max()
                    .sort_values(ascending=False)
                    .index.tolist()
                )

                expanded["IssueType"] = expanded["IssueType"].fillna("Unknown")
                grouped = expanded.groupby(["Sprint", "IssueType"]).size().reset_index(name="Count")
                totals = expanded.groupby("Sprint").size().reset_index(name="Total")
                merged = pd.merge(grouped, totals, on="Sprint", how="left")
                merged["Percent"] = (merged["Count"] / merged["Total"] * 100)

                # Fixed order for compact summary
                ORDER = {"Story": 0, "Bug": 1, "Spike": 2, "Task": 3}
                def make_compact(group):
                    d = dict(zip(group["IssueType"], group["Percent"]))
                    parts = []
                    for it in ["Story", "Bug", "Spike", "Task"]:
                        if it in d:
                            parts.append(f"{it}: {d[it]:.0f}%")
                    # append any other types, alphabetically
                    extras = sorted([k for k in d.keys() if k not in ORDER])
                    for it in extras:
                        parts.append(f"{it}: {d[it]:.0f}%")
                    return ", ".join(parts)

                compact = (
                    merged.groupby("Sprint")
                          .apply(make_compact)
                          .rename("Composition")
                          .reindex(sprint_order)
                )

                st.markdown("**Sprint breakdown — click a sprint to expand**")
                for spr in sprint_order:
                    label = f"{spr} — {compact.get(spr, '') if hasattr(compact, 'get') else compact.loc[spr] if spr in compact.index else ''}"
                    with st.expander(label):
                        spr_total = int(totals[totals["Sprint"] == spr]["Total"].iloc[0]) if spr in totals["Sprint"].values else 0
                        st.caption(f"Total items: {spr_total}")

                        g = merged[merged["Sprint"] == spr].copy()
                        # Sort rows in the drilldown by the same fixed order (others at the end A-Z)
                        g["order_key"] = g["IssueType"].map(ORDER).fillna(99)
                        g = g.sort_values(["order_key", "IssueType"]).drop(columns=["order_key"])
                        g["Percent"] = g["Percent"].round(1)
                        st.dataframe(g[["IssueType", "Count", "Percent"]].reset_index(drop=True), use_container_width=True)
