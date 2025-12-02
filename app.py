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


# ---- Layout tweaks ----
st.set_page_config(page_title="Cycle Time Analysis (Timepiece)", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===================== SIDEBAR: TIMEPIECE CONFIG =====================
with st.sidebar:
    st.title("Timepiece Settings")
    st.caption("OBSS Timepiece Cloud → Jira")

    # NOTE: In real use, move this to secrets
    api_key = st.text_input(
        "API Key",
        type="password",
        value="",
        help="OBSS Timepiece API key",
    )

    filter_id = st.text_input(
        "Saved Filter ID",
        value="",
        help="Jira filter ID used by Timepiece",
    )

    st.markdown("### Status Buckets")
    default_rules = """Blocked = On Hold (C7SM)
Development = In Development (C7SM), In Progress (C7O), In Development (C7T4)
Review = Review (C7SM)"""
    rules_text = st.text_area(
        "Rules (Bucket = Status1, Status2, ...)", value=default_rules, height=120
    )

    fetch_button = st.button("Fetch Data")

if not api_key or not filter_id:
    st.info("Enter API Key and Filter ID, then press **Fetch Data**.")
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
            st.success(f"Fetched {len(raw)} issues from Timepiece.")
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()
elif "raw_data" not in st.session_state:
    st.info("Press **Fetch Data** to load data from Timepiece.")
    st.stop()
else:
    raw = st.session_state["raw_data"]


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
                sampled = np.random.choice(completions_per_week)
                total_done += sampled
                week_count += 1
            results.append(week_count)

    elif weeks:  # Forecast number of items done in X weeks
        for _ in range(num_simulations):
            total_done = 0
            for _ in range(weeks):
                sampled = np.random.choice(completions_per_week)
                total_done += sampled
            results.append(total_done)

    return np.array(results)


# ===================== MAIN CONTROLS =====================
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
tabs = st.tabs(["Cycle Time", "Slowest Items", "Forecasting", "Data", "Sprint Analysis", "Context"])


# ---------- OVERVIEW ----------
with tabs[0]:

    # ---- Merged Overview & Insights (view-only) ----
    st.subheader("Cycle Time — Overview & Insights" + ("" if selected_team == "All Teams" else f" — {selected_team}"))
    col1, col2, col3, col4, col5 = st.columns(5)

    avg_ct = round(view_df["CT"].mean(), 2) if "CT" in view_df.columns and view_df["CT"].notna().any() else np.nan
    p85_ct = round(view_df["CT"].quantile(0.85), 2) if "CT" in view_df.columns and view_df["CT"].notna().any() else np.nan

    if "_bucketer" in view_df.columns and view_df["_bucketer"].notna().any():
        _vc = (
            view_df["_bucketer"].dropna().dt.to_period("M").value_counts().sort_index()
        )
        items_this_month = int(_vc.iloc[-1]) if len(_vc) else 0
    else:
        items_this_month = 0

    col1.metric("Average CT (overall)", "--" if np.isnan(avg_ct) else avg_ct)
    col2.metric("85th Percentile CT (overall)", "--" if np.isnan(p85_ct) else p85_ct)
    col3.metric("Items this month", items_this_month)

    st.divider()
    
    
    with tabs[0]:
        st.subheader("Cycle Time Trends")

    
        # === Enhanced CT Trends (Monthly/Rolling, Metric selector incl. P85, counts, provisional month) ===
        def _build_ct_series_from_view(df: pd.DataFrame, tz: str = "Europe/London", rolling_days: int = 30, min_n_for_pct: int = 1):
            if df is None or df.empty or "End" not in df.columns or "CT" not in df.columns:
                return {"monthly": pd.DataFrame(), "rolling": pd.DataFrame()}
            cols = ["End", "CT"] + (["IssueType"] if "IssueType" in df.columns else [])
            d = df[cols].dropna(subset=["End", "CT"]).copy()
            d.rename(columns={"End": "completed_date", "CT": "cycle_time_days"}, inplace=True)
            d["completed_date"] = pd.to_datetime(d["completed_date"], utc=True, errors="coerce")
            d = d.dropna(subset=["completed_date", "cycle_time_days"])
            d["completed_date"] = d["completed_date"].dt.tz_convert(tz)
    
            # Monthly
            d["month"] = (
                d["completed_date"]
                .dt.to_period("M")
                .dt.to_timestamp("M")  # month-end for clearer labelling
                .dt.tz_localize(tz)
            )
            monthly = (
                d.groupby("month", as_index=False)
                 .agg(mean_ct=("cycle_time_days", "mean"),
                      median_ct=("cycle_time_days", "median"),
                      p85_ct=("cycle_time_days", lambda x: x.quantile(0.85)),
                      count=("cycle_time_days", "size"))
                 .sort_values("month")
                 .reset_index(drop=True)
            )
            now_tz = pd.Timestamp.now(tz=tz)

            # Use end-of-month (to match the month-end we store in `monthly["month"]`)
            current_month_end = (
                pd.Timestamp(year=now_tz.year, month=now_tz.month, day=1, tz=now_tz.tz)
                + pd.offsets.MonthEnd(0)
            )

            monthly["is_current_month"] = monthly["month"] == current_month_end
    
            # Monthly type breakdown for tooltip
            if "IssueType" in d.columns:
                mt = (
                    d.groupby([
                    d["completed_date"]
                    .dt.to_period("M")
                    .dt.to_timestamp("M")
                    .dt.tz_localize(tz),
                    "IssueType",
                ])
                     .size()
                     .unstack(fill_value=0)
                )
                mt = mt.reindex(monthly["month"]).fillna(0).astype(int)
            else:
                mt = None
            
            if mt is not None:
                monthly["type_breakdown"] = mt.apply(
                    lambda row: ", ".join([f"{col}: {val}" for col, val in row.items() if val > 0]),
                    axis=1,
                )
            else:
                monthly["type_breakdown"] = ""
            
            # Rolling window
            if rolling_days <= 1:
                return {"monthly": monthly, "rolling": pd.DataFrame()}
            start_date = d["completed_date"].min().floor("D")
            end_date = d["completed_date"].max().ceil("D")
            day_index = pd.date_range(start=start_date, end=end_date, freq="D", tz=tz)
    
            means, medians, p85s, counts, types_info = [], [], [], [], []
            for day in day_index:
                window_start = day - pd.Timedelta(days=rolling_days - 1)
                mask = (d["completed_date"] >= window_start) & (d["completed_date"] <= day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
                vals = d.loc[mask, "cycle_time_days"]
                c = int(vals.shape[0]); counts.append(c)
                means.append(float(vals.mean()) if c else None)
                medians.append(float(vals.median()) if c else None)
                p85s.append(float(vals.quantile(0.85)) if c > 0 else None)
           
                if "IssueType" in d.columns and c >= min_n_for_pct:
                    tcounts = d.loc[mask, "IssueType"].value_counts()
                    info = ", ".join([f"{idx}: {cnt}" for idx, cnt in tcounts.items()])
                else:
                    info = ""
                types_info.append(info)
    
            rolling = pd.DataFrame(
                {
                    "date": day_index,
                    "mean_ct": means,
                    "median_ct": medians,
                    "p85_ct": p85s,
                    "count": counts,
                    "type_breakdown": types_info,
                }
            ).dropna(subset=["median_ct"])
    
            return {"monthly": monthly, "rolling": rolling}
    
        if view_df.empty:
            st.info("No rows match the current filters.")
        else:
            ts_choice = st.radio(
                "Trend View",
                ["Monthly", "Rolling (30d)", "Rolling (14d)"],
                index=0,
                horizontal=True,
            )
            metric_choice = st.selectbox("Metric", ["Median (P50)", "Average (Mean)", "P85"], index=0)
            include_current_month = st.checkbox("Include current month (monthly view)", value=False, key="ctenh_current", help="If off, current month is excluded from monthly chart to avoid early-month bias.")
    
        ewindow = 30 if eview.startswith("Rolling (30") else 14 if eview.startswith("Rolling (14") else 30
        ets = _build_ct_series_from_view(view_df, rolling_days=ewindow)
    
        # Single KPI tile based on selected Metric (last 30d)
        ets30 = _build_ct_series_from_view(view_df, rolling_days=30)
        last30 = ets30.get("rolling", pd.DataFrame())
        metric_map = {"Median (P50)": ("Median CT (30d)", "median_ct"),
                      "Average (Mean)": ("Avg CT (30d)", "mean_ct"),
                      "P85": ("P85 CT (30d)", "p85_ct")}
        
        label, field = metric_map.get(metric_choice, ("Median CT (30d)", "median_ct"))
        if not last30.empty:
            recent = last30.tail(30)  # last 30 days
            val = recent[field].dropna().iloc[-1] if recent[field].notna().any() else np.nan
        else:
            val = np.nan
    
        col4.metric(label, "--" if np.isnan(val) else round(val, 2))
    
        # Build chart
        if ts_choice.startswith("Monthly"):
            monthly = ets.get("monthly", pd.DataFrame())
            if not include_current_month and "is_current_month" in monthly.columns:
                monthly = monthly[~monthly["is_current_month"]]
    
            if monthly.empty:
                st.info("No data for monthly trends.")
            else:
                metric_field = {"Median (P50)": "median_ct", "Average (Mean)": "mean_ct", "P85": "p85_ct"}[metric_choice]
                ytitle = f"{metric_choice} CT (days)"
    
                monthly["month_label"] = monthly["month"].dt.strftime("%Y-%m")
    
                chart = (
                    alt.Chart(monthly)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("month_label:N", title="Month"),
                        y=alt.Y(f"{metric_field}:Q", title=ytitle),
                        tooltip=[
                            alt.Tooltip("month_label:N", title="Month"),
                            alt.Tooltip(f"{metric_field}:Q", title="CT (days)", format=".2f"),
                            alt.Tooltip("count:Q", title="# Items"),
                            alt.Tooltip("type_breakdown:N", title="Types"),
                        ],
                    )
                )
                st.altair_chart(chart, use_container_width=True)
    
        else:
            rolling = ets.get("rolling", pd.DataFrame())
            if rolling.empty:
                st.info("Not enough data for rolling trends.")
            else:
                metric_field = {"Median (P50)": "median_ct", "Average (Mean)": "mean_ct", "P85": "p85_ct"}[metric_choice]
                ytitle = f"{metric_choice} CT (days, window={ewindow}d)"
    
                rolling["date_label"] = rolling["date"].dt.strftime("%Y-%m-%d")
    
                chart = (
                    alt.Chart(rolling)
                    .mark_line()
                    .encode(
                        x=alt.X("date_label:N", title="Date"),
                        y=alt.Y(f"{metric_field}:Q", title=ytitle),
                        tooltip=[
                            alt.Tooltip("date_label:N", title="Date"),
                            alt.Tooltip(f"{metric_field}:Q", title="CT (days)", format=".2f"),
                            alt.Tooltip("count:Q", title="# Items in window"),
                            alt.Tooltip("type_breakdown:N", title="Types in window"),
                        ],
                    )
                )
                st.altair_chart(chart, use_container_width=True)


# ---------- SLOWEST ITEMS ----------
with tabs[1]:
    st.subheader("Slowest Items")

    if view_df.empty:
        st.info("No data in current filter set.")
    else:
        top_n = st.slider("How many items to show?", 5, 100, 20)
        slowest = view_df.sort_values("CT", ascending=False).head(top_n)

        cols_to_show = ["Key", "Summary", "IssueType", "CT", "End"]
        cols_to_show = [c for c in cols_to_show if c in slowest.columns]

        st.dataframe(slowest[cols_to_show], use_container_width=True)


# ---------- FORECASTING ----------
with tabs[2]:
    st.subheader("Forecasting (Throughput-based)")

    if view_df.empty or "End" not in view_df.columns:
        st.info("No data to forecast from.")
    else:
        completed = view_df.dropna(subset=["End"]).copy()
        completed["End"] = pd.to_datetime(completed["End"], errors="coerce")
        completed["week"] = completed["End"].dt.to_period("W").apply(lambda r: r.start_time)

        throughput = completed.groupby("week")["Key"].count().reset_index(name="Completed")
        if throughput.empty:
            st.info("No completed items to forecast from.")
        else:
            st.write("Throughput per week:")
            st.bar_chart(throughput.set_index("week")["Completed"])

            completions = throughput["Completed"].values

            mode = st.radio("Forecast mode", ["How many weeks for X items?", "How many items in X weeks?"], index=0)

            num_simulations = st.slider("Number of simulations", 1000, 20000, 5000, step=1000)

            if mode == "How many weeks for X items?":
                items = st.number_input("Number of items", min_value=1, value=10)
                if st.button("Run forecast"):
                    sims = monte_carlo_forecast_throughput(completions, num_simulations=num_simulations, items=items)
                    p50 = np.percentile(sims, 50)
                    p85 = np.percentile(sims, 85)
                    p95 = np.percentile(sims, 95)

                    st.metric("P50 (weeks)", round(p50, 1))
                    st.metric("P85 (weeks)", round(p85, 1))
                    st.metric("P95 (weeks)", round(p95, 1))
            else:
                weeks = st.number_input("Number of weeks", min_value=1, value=10)
                if st.button("Run forecast"):
                    sims = monte_carlo_forecast_throughput(completions, num_simulations=num_simulations, weeks=weeks)
                    p50 = np.percentile(sims, 50)
                    p85 = np.percentile(sims, 85)
                    p95 = np.percentile(sims, 95)

                    st.metric("P50 (items)", round(p50, 1))
                    st.metric("P85 (items)", round(p85, 1))
                    st.metric("P95 (items)", round(p95, 1))


# ---------- DATA ----------
with tabs[3]:
    st.subheader("Data preview")

    row_count = len(view_df)
    st.caption(f"Showing all {row_count} rows returned from Timepiece")

    st.dataframe(view_df, use_container_width=True)


# ---------- CONTEXT / LEADERSHIP VIEW ----------
with tabs[5]:
    st.subheader("Business Context & Leadership View")
    st.caption(
        "Use this tab to line up cycle time with people movements, big projects, and other constraints "
        "without using it to judge team performance."
    )

    st.markdown(
        """
        **How to use this tab**

        1. Prepare a CSV of context events with at least these columns (case-insensitive):
           - `team` (or `squad`)
           - `start_date` (or `start`)
        2. Optional columns (if present, they will be used in tooltips and the timeline):
           - `end_date` / `end` / `finish`
           - `event_type` / `type` / `category`
           - `who` / `person` / `people` / `owner`
           - `description` / `details` / `note` / `notes`
        3. Upload the CSV below and select a team at the top of the page.
        """
    )

    ctx_file = st.file_uploader(
        "Upload context events CSV",
        type="csv",
        key="context_events_csv",
        help="Contains key people moves, project phases, holidays, etc."
    )

    context_df = None
    if ctx_file is not None:
        try:
            raw_ctx = pd.read_csv(ctx_file)
            if raw_ctx.empty:
                st.warning("The uploaded CSV appears to be empty.")
            else:
                # Normalise column names
                lower_map = {c.lower().strip(): c for c in raw_ctx.columns}

                def _col(options):
                    for o in options:
                        if o in lower_map:
                            return lower_map[o]
                    return None

                team_col_ctx = _col(["team", "squad"])
                start_col = _col(["start_date", "start"])
                end_col = _col(["end_date", "end", "finish"])
                type_col = _col(["event_type", "type", "category"])
                who_col = _col(["who", "person", "people", "owner"])
                desc_col = _col(["description", "details", "note", "notes"])

                required = [team_col_ctx, start_col]
                if any(c is None for c in required):
                    st.error(
                        "CSV must include at least 'team' and 'start_date' columns "
                        "(we also accept 'squad' and 'start')."
                    )
                else:
                    context_df = pd.DataFrame()
                    context_df["Team"] = raw_ctx[team_col_ctx].astype(str)

                    context_df["Start"] = pd.to_datetime(
                        raw_ctx[start_col], errors="coerce"
                    )
                    if end_col:
                        context_df["End"] = pd.to_datetime(
                            raw_ctx[end_col], errors="coerce"
                        )
                    else:
                        context_df["End"] = pd.NaT

                    if type_col:
                        context_df["EventType"] = raw_ctx[type_col].astype(str)
                    else:
                        context_df["EventType"] = "Event"

                    if who_col:
                        context_df["Who"] = raw_ctx[who_col].astype(str)
                    else:
                        context_df["Who"] = ""

                    if desc_col:
                        context_df["Description"] = raw_ctx[desc_col].astype(str)
                    else:
                        context_df["Description"] = ""

                    # Drop rows with no valid Start date
                    context_df = context_df.dropna(subset=["Start"])
                    if context_df.empty:
                        st.warning(
                            "No valid rows after parsing dates. "
                            "Check that your start_date column contains real dates."
                        )
        except Exception as e:
            st.error(f"Could not read context CSV: {e}")

    if context_df is None:
        st.info(
            "Upload a context-events CSV to see overlays and summaries. "
            "Expected columns at minimum: Team, Start_date; optional: End_date, EventType, Who, Description."
        )
    else:
        st.markdown("**Context events loaded (preview)**")
        st.dataframe(context_df.head(50), use_container_width=True)

        st.divider()

        # --- 1. CT vs Time with context overlays for the selected team ---
        st.markdown("### 1. Cycle time with context overlays")

        if selected_team == "All Teams":
            st.info(
                "Select a specific team at the top of the page to see cycle time with context overlays."
            )
        else:
            team_ctx = context_df[context_df["Team"] == selected_team].copy()

            if team_ctx.empty:
                st.info(f"No context events found for {selected_team}.")
            else:
                # Use the same helper used in the CT tab to build monthly series
                try:
                    series = _build_ct_series_from_view(view_df)
                    monthly = series.get("monthly", pd.DataFrame())
                except NameError:
                    monthly = pd.DataFrame()

                if monthly.empty:
                    st.info("No completed items to plot for this team.")
                else:
                    base = alt.Chart(monthly).encode(
                        x=alt.X("month:T", title="Month")
                    )

                    # Median CT line
                    line = base.mark_line(point=True).encode(
                        y=alt.Y("median_ct:Q", title="Median CT (days)"),
                        tooltip=[
                            alt.Tooltip("month:T", title="Month"),
                            alt.Tooltip("median_ct:Q", title="Median CT", format=".2f"),
                            alt.Tooltip("p85_ct:Q", title="P85 CT", format=".2f"),
                            alt.Tooltip("count:Q", title="Items"),
                        ],
                    )

                    # Event markers: vertical dashed rules at Start
                    event_chart = (
                        alt.Chart(team_ctx)
                        .mark_rule(strokeDash=[4, 4], size=2)
                        .encode(
                            x=alt.X("Start:T", title=""),
                            color=alt.Color("EventType:N", title="Event"),
                            tooltip=[
                                alt.Tooltip("EventType:N", title="Event"),
                                alt.Tooltip("Who:N", title="Who"),
                                alt.Tooltip("Description:N", title="Details"),
                                alt.Tooltip("Start:T", title="Start"),
                                alt.Tooltip("End:T", title="End"),
                            ],
                        )
                    )

                    st.altair_chart(
                        (line + event_chart).properties(height=380),
                        use_container_width=True,
                    )

                    # Human-readable timeline
                    st.markdown("#### Context timeline")
                    team_ctx_sorted = team_ctx.sort_values("Start")
                    for _, row in team_ctx_sorted.iterrows():
                        date_str = row["Start"].date().isoformat()
                        title = f"**{date_str} — {row['EventType']}**"
                        who = (
                            f" ({row['Who']})"
                            if isinstance(row["Who"], str)
                            and row["Who"].strip()
                            else ""
                        )
                        desc = (
                            f"{row['Description']}"
                            if isinstance(row["Description"], str)
                            else ""
                        )
                        st.markdown(
                            f"{title}{who}<br/>{desc}",
                            unsafe_allow_html=True,
                        )

        st.divider()

        # --- 2. Context load heatmap across teams ---
        st.markdown("### 2. Context load heatmap (all teams)")

        ctx_heat = context_df.copy()
        ctx_heat["Month"] = (
            ctx_heat["Start"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        heat = (
            ctx_heat.groupby(["Team", "Month"])
            .size()
            .reset_index(name="EventCount")
        )

        if heat.empty:
            st.info("No context events to show in heatmap.")
        else:
            heat_chart = (
                alt.Chart(heat)
                .mark_rect()
                .encode(
                    x=alt.X("Month:T", title="Month"),
                    y=alt.Y("Team:N", title="Team"),
                    color=alt.Color("EventCount:Q", title="Events"),
                    tooltip=[
                        alt.Tooltip("Team:N", title="Team"),
                        alt.Tooltip("Month:T", title="Month"),
                        alt.Tooltip("EventCount:Q", title="Events"),
                    ],
                )
            )

            st.altair_chart(
                heat_chart.properties(height=260),
                use_container_width=True,
            )


# ---------- SPRINT ANALYSIS ----------
try:
    # Index 4 == "Sprint Analysis" (we added 'Context' as index 5)
    sprint_tab = tabs[4]
except Exception:
    sprint_tab = None

if sprint_tab is not None:
    with sprint_tab:
                st.subheader("Sprint Analysis")

    df_src = view_df if 'view_df' in globals() else (df if 'df' in globals() else None)
    if df_src is None or df_src.empty or "Sprint" not in df_src.columns or "IssueType" not in df_src.columns:
        st.info("No sprint/issue type data available.")
    else:
        # Split multi-sprint entries
        expanded = df_src.copy()
        expanded["Sprint"] = expanded["Sprint"].fillna("")
        expanded = expanded.assign(Sprint=expanded["Sprint"].astype(str).str.split(","))
        expanded = expanded.explode("Sprint")
        expanded["Sprint"] = expanded["Sprint"].astype(str).str.strip()
        expanded = expanded[expanded["Sprint"] != ""]

        if expanded.empty:
            st.info("No sprint data after expansion.")
        else:
            # Convert End dates for ordering
            end_col = "End" if "End" in expanded.columns else ("Resolved" if "Resolved" in expanded.columns else None)
            expanded["_enddate"] = pd.to_datetime(expanded[end_col], errors="coerce") if end_col else pd.NaT

            # Order sprints from latest to oldest
            sprint_order = (
                expanded.groupby("Sprint")["_enddate"]
                .max()
                .sort_values(ascending=False)
                .index.tolist()
            )

            expanded["IssueType"] = expanded["IssueType"].fillna("Unknown")

            # Basic counts
            grouped = expanded.groupby(["Sprint", "IssueType"]).size().reset_index(name="Count")
            totals = expanded.groupby("Sprint").size().reset_index(name="Total")
            merged = pd.merge(grouped, totals, on="Sprint", how="left")
            merged["Percent"] = merged["Count"] / merged["Total"] * 100

            merged["Sprint"] = pd.Categorical(merged["Sprint"], categories=sprint_order, ordered=True)
            merged = merged.sort_values("Sprint")

            if sprint_tab is not None:
                with sprint_tab:
                    st.markdown("### Sprint Composition by Issue Type")
                    comp_chart = (
                        alt.Chart(merged)
                        .mark_bar()
                        .encode(
                            x=alt.X("Sprint:N", sort=sprint_order, title="Sprint"),
                            y=alt.Y("Percent:Q", stack="normalize", title="Proportion"),
                            color=alt.Color("IssueType:N", title="Issue Type"),
                            tooltip=[
                                alt.Tooltip("Sprint:N", title="Sprint"),
                                alt.Tooltip("IssueType:N", title="Issue Type"),
                                alt.Tooltip("Count:Q", title="Count"),
                                alt.Tooltip("Percent:Q", title="% of Sprint", format=".1f"),
                            ],
                        )
                    )
                    st.altair_chart(comp_chart.properties(height=320), use_container_width=True)
