import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys, os
import datetime

# ---- Fibonacci Overlay Chart ----
import matplotlib.pyplot as plt
import numpy as np

def fib_overlay_chart(dates, values, title="P85 Cycle Time with Fibonacci Bands"):
    values = np.array(values, dtype=float)
    low = np.nanmin(values)
    high = np.nanmax(values)

    # Fibonacci ratios (flipped: lower = better)
    fib = np.array([1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0])
    fib_vals = low + (high - low) * fib

    fig, ax = plt.subplots(figsize=(14, 4))

    # Shaded zones
    ax.axhspan(fib_vals[2], fib_vals[0], color="#ffebee", alpha=0.35)   # red zone (worst)
    ax.axhspan(fib_vals[3], fib_vals[2], color="#fff8e1", alpha=0.35)   # orange
    ax.axhspan(fib_vals[4], fib_vals[3], color="#f1f8e9", alpha=0.35)   # yellow/green
    ax.axhspan(fib_vals[-1], fib_vals[4], color="#e8f5e9", alpha=0.35)  # green (best)

    # Fibonacci dashed lines
    for lvl in fib_vals:
        ax.axhline(lvl, linestyle="--", linewidth=0.7, color="gray")

    # Main CT line
    ax.plot(dates, values, marker="o", color="#00AEEF", linewidth=2)

    ax.set_title(title)
    ax.set_ylabel("P85 CT (days)")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(
        [d.strftime("%b %d") if hasattr(d, "strftime") else str(d) for d in dates],
        rotation=45
    )

    plt.tight_layout()
    return fig


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
tabs = st.tabs([
    "Cycle Time",
    "Slowest Items",
    "Forecasting",
    "Data",
    "Sprint Analysis",
    "Context"     # <- new tab
])

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
                monthly["type_breakdown"] = mt.apply(lambda row: ", ".join([f"{k}: {int(v)}" for k, v in row.items() if v > 0]) if row.sum() > 0 else "", axis=1)
            else:
                monthly["type_breakdown"] = ""
    
            # Rolling trailing window
            if d.empty:
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
                if "IssueType" in d.columns:
                    sub = d.loc[mask, ["IssueType"]]
                    if not sub.empty:
                        cnts = sub.value_counts().to_dict()
                        types_info.append(", ".join([f"{k}: {v}" for k, v in cnts.items()]))
                    else:
                        types_info.append("")
                else:
                    types_info.append("")
    
            rolling = pd.DataFrame({"date": day_index, "mean_ct": means, "median_ct": medians, "p85_ct": p85s, "count": counts, "types_breakdown": types_info})
            return {"monthly": monthly, "rolling": rolling}
    
        # Controls
        ec1, ec2, ec3, ec4 = st.columns([1.3, 1.2, 1, 1.35])
        eview = ec1.selectbox("Time grouping", ["Monthly", "Rolling (30 days)", "Rolling (14 days)"], index=0, key="ctenh_view")
        metric = ec2.selectbox("Metric", ["Median (P50)", "Average (Mean)", "P85"], index=0, key="ctenh_metric")
        eshow_counts = ec3.checkbox("Show item counts", value=True, key="ctenh_counts")
        einclude_current = ec4.checkbox("Include current month", value=False, key="ctenh_current", help="If off, current month is excluded from monthly chart to avoid early-month bias.")
    
        ewindow = 30 if eview.startswith("Rolling (30") else 14 if eview.startswith("Rolling (14") else 30
        ets = _build_ct_series_from_view(view_df, rolling_days=ewindow)
    
        # Single KPI tile based on selected Metric (last 30d)
        ets30 = _build_ct_series_from_view(view_df, rolling_days=30)
        last30 = ets30.get("rolling", pd.DataFrame())
        metric_map = {"Median (P50)": ("Median CT (30d)", "median_ct"),
                      "Average (Mean)": ("Avg CT (30d)", "mean_ct"),
                      "P85": ("P85 CT (30d)", "p85_ct")}
        label, field = metric_map.get(metric, ("Median CT (30d)", "median_ct"))
        value = None
        if last30 is not None and not last30.empty and field in last30.columns:
            series = last30[field].dropna()
            if not series.empty:
                value = float(series.iloc[-1])
        st.metric(label, f"{value:.1f} days" if value is not None else "—")
    
        # Per-issue-type tiles (Story/Bug/Task/Spike) for last 30d
        try:
            if "IssueType" in view_df.columns and "CT" in view_df.columns and "End" in view_df.columns:
                _d = view_df.copy()
                _d["End"] = pd.to_datetime(_d["End"], utc=True, errors="coerce")
                if not _d["End"].isna().all():
                    _end = _d["End"].max()
                    _start = _end - pd.Timedelta(days=30)
                    sub = _d.loc[(_d["End"] >= _start) & (_d["End"] <= _end), ["IssueType", "CT"]].dropna()
                    if not sub.empty:
                        if metric == "Median (P50)":
                            agg = sub.groupby("IssueType")["CT"].median().round(1)
                        elif metric == "Average (Mean)":
                            agg = sub.groupby("IssueType")["CT"].mean().round(1)
                        else:
                            agg = sub.groupby("IssueType")["CT"].quantile(0.85).round(1)
                        order = ["Story", "Bug", "Task", "Spike"]
                        cols = st.columns(4)
                        for i, it in enumerate(order):
                            cols[i].metric(f"{it} (30d)", f"{agg[it]} days" if it in agg.index else "—")
        except Exception:
            pass
    
        # Chart
        y_field_map = {"Median (P50)": "median_ct", "Average (Mean)": "mean_ct", "P85": "p85_ct"}
        y_field = y_field_map[metric]
        y_title = f"{metric} CT (days)"
    
        if eview == "Monthly":
            if eview == "Monthly":
                use_fib_overlay = st.checkbox("Show Fibonacci Performance Bands (P85 only)", value=False)

            monthly = ets["monthly"]
            if monthly.empty:
                st.info("No completed items to plot.")
            else:
                if not einclude_current and "is_current_month" in monthly.columns:
                    monthly = monthly[~monthly["is_current_month"]]
    
                base = alt.Chart(monthly).encode(x=alt.X("month:T", title="Month"))
                line = base.mark_line(point=True).encode(
                    y=alt.Y(f"{y_field}:Q", title=y_title),
                    tooltip=[
                        alt.Tooltip("month:T", title="Month"),
                        alt.Tooltip(f"{y_field}:Q", title=y_title, format=".2f"),
                        alt.Tooltip("count:Q", title="Items"),
                        alt.Tooltip("type_breakdown:N", title="Types"),
                    ],
                )
                chart = line
                if einclude_current and "is_current_month" in ets["monthly"].columns:
                    provisional = base.transform_filter(alt.datum.is_current_month == True).mark_point(size=120, opacity=0.2).encode(
                        y=f"{y_field}:Q",
                        tooltip=[alt.Tooltip("month:T", title="Provisional Month")],
                    )
                    chart = (line + provisional)
    
                if eshow_counts:
                    bars = base.mark_bar(opacity=0.2).encode(y=alt.Y("count:Q", title="Items (count)", axis=alt.Axis(titleColor="#666")))
                    labels = base.mark_text(dy=-18).encode(text="count:Q")
                    chart = (bars + line + labels)
    
                # --- Fibonacci override for P85 Monthly view ---
                if use_fib_overlay and metric == "P85":
                
                    # Aggregate monthly values properly (CRITICAL FIX)
                    df_plot = (
                        monthly.dropna(subset=["p85_ct"])
                               .groupby("month", as_index=False)
                               .agg({"p85_ct": "mean"})
                    )
                
                    # Convert month values -> datetime
                    dates = [pd.to_datetime(str(m)) for m in df_plot["month"].tolist()]
                
                    vals = df_plot["p85_ct"].tolist()
                
                    fig = fib_overlay_chart(dates, vals, title=f"{selected_team} — P85 Fibonacci Trend")
                    st.pyplot(fig)
                
                else:
                    st.altair_chart(chart.properties(height=380), use_container_width=True)


        else:
            rolling = ets["rolling"]
            if rolling.empty:
                st.info("No completed items to plot.")
            else:
                base = alt.Chart(rolling).encode(x=alt.X("date:T", title="Date"))
                line = base.mark_line().encode(
                    y=alt.Y(f"{y_field}:Q", title=f"{y_title} — trailing {ewindow}d"),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip(f"{y_field}:Q", title=y_title, format=".2f"),
                        alt.Tooltip("count:Q", title=f"Items in last {ewindow}d"),
                        alt.Tooltip("types_breakdown:N", title="Types"),
                    ],
                )
                chart = line
                if eshow_counts:
                    area = base.mark_area(opacity=0.15).encode(
                        y=alt.Y("count:Q", title=f"Items (last {ewindow}d)"),
                        tooltip=[alt.Tooltip("count:Q", title=f"Items (last {ewindow}d)")],
                    )
                    chart = (area + line)
                st.altair_chart(chart.properties(height=380), use_container_width=True)


with tabs[1]:
    # ---------- SLOWEST ITEMS ----------
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


with tabs[2]:
    # ---------- FORECASTING ----------
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
                    st.altair_chart(hist, use_container_width=True)

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
                    st.altair_chart(hist, use_container_width=True)


# ---------- DATA ----------
with tabs[3]:
    st.subheader("Data preview")

    row_count = len(view_df)
    st.caption(f"Showing all {row_count} rows returned from Timepiece")

    st.dataframe(view_df, use_container_width=True)


# ---------- SPRINT ANALYSIS ----------
try:
    sprint_tab = tabs[4]   # index 4 = Sprint Analysis
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
                merged["Percent"] = (merged["Count"] / merged["Total"] * 100)

                # Issue type ordering
                ORDER = {"Story": 0, "Bug": 1, "Spike": 2, "Task": 3}

                def make_compact(group):
                    d = dict(zip(group["IssueType"], group["Percent"]))
                    parts = []
                    for it in ["Story", "Bug", "Spike", "Task"]:
                        if it in d:
                            parts.append(f"{it}: {d[it]:.0f}%")
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
                    sprint_items = expanded[expanded["Sprint"] == spr]
                    if not sprint_items.empty and "CT" in sprint_items.columns:
                        sprint_p85 = sprint_items["CT"].quantile(0.85)
                        sprint_p85_text = f" • 85th CT: {sprint_p85:.1f} days"
                    else:
                        sprint_p85_text = ""

                    label = f"{spr} — {compact.get(spr, '')}{sprint_p85_text}"

                    with st.expander(label):
                        spr_total = int(totals[totals["Sprint"] == spr]["Total"].iloc[0]) if spr in totals["Sprint"].values else 0
                        st.caption(f"Total items: {spr_total}")

                        g = merged[merged["Sprint"] == spr].copy()
                        g["order_key"] = g["IssueType"].map(ORDER).fillna(99)
                        g = g.sort_values(["order_key", "IssueType"]).drop(columns=["order_key"])
                        g["Percent"] = g["Percent"].round(1)

                        if not sprint_items.empty and "CT" in sprint_items.columns:
                            p85_by_type = sprint_items.groupby("IssueType")["CT"].quantile(0.85)
                        else:
                            p85_by_type = {}

                        g["85th CT (days)"] = g["IssueType"].map(p85_by_type).round(2)

                        st.dataframe(
                            g[["IssueType", "Count", "Percent", "85th CT (days)"]].reset_index(drop=True),
                            use_container_width=True,
                        )

                        st.markdown("**Items by Issue Type**")

                        has_key_col = "Key" in sprint_items.columns
                        has_summary_col = "Summary" in sprint_items.columns

                        for itype in g["IssueType"].unique():
                            sub_items = sprint_items[sprint_items["IssueType"] == itype].copy()
                            if sub_items.empty:
                                continue

                            title = f"{itype} ({len(sub_items)} items)"
                            with st.expander(title):
                                cols_to_show = []
                                if has_key_col:
                                    cols_to_show.append("Key")
                                if has_summary_col:
                                    cols_to_show.append("Summary")
                                for col in ["IssueType", "CT", "Start", "End"]:
                                    if col in sub_items.columns and col not in cols_to_show:
                                        cols_to_show.append(col)

                                if cols_to_show:
                                    st.dataframe(
                                        sub_items[cols_to_show]
                                        .sort_values("CT", ascending=False)
                                        .reset_index(drop=True),
                                        use_container_width=True,
                                    )
                                else:
                                    st.info("No item-level columns available to display for this sprint.")


# ====================== CONTEXT / LEADERSHIP TAB ======================
with tabs[5]:
    st.subheader("Business Context & Leadership View")
    st.caption("Overlay cycle time with business context like people moves, holidays, and major project periods.")

    st.markdown("""
    **How to use this section**
    1. Create a CSV with at least:
       - `team`
       - `start_date`
    2. Optional columns:
       - `end_date`
       - `event_type`
       - `who`
       - `description`
    3. Upload it below.
    4. Select a team at the top of the app.
    """)

    ctx_file = st.file_uploader("Upload context events CSV", type="csv")

    if ctx_file is None:
        st.info("Upload a CSV to display context overlays.")
    else:
        # Load CSV
        ctx = pd.read_csv(ctx_file)

        # Normalise lower case
        cols = {c.lower().strip(): c for c in ctx.columns}

        def find_col(options):
            for opt in options:
                if opt in cols:
                    return cols[opt]
            return None

        team_col = find_col(["team", "squad"])
        start_col = find_col(["start_date", "start"])
        end_col = find_col(["end_date", "end"])
        type_col = find_col(["event_type", "type"])
        who_col = find_col(["who", "person"])
        desc_col = find_col(["description", "details", "note"])

        if team_col is None or start_col is None:
            st.error("CSV must contain at least 'team' and 'start_date' columns.")
            st.stop()

        # Build context DF
        ctx_df = pd.DataFrame()
        ctx_df["Team"] = ctx[team_col].astype(str)
        ctx_df["Start"] = pd.to_datetime(ctx[start_col], errors="coerce")
        ctx_df["End"] = pd.to_datetime(ctx[end_col], errors="coerce") if end_col else pd.NaT
        ctx_df["EventType"] = ctx[type_col].astype(str) if type_col else "Event"
        ctx_df["Who"] = ctx[who_col].astype(str) if who_col else ""
        ctx_df["Description"] = ctx[desc_col].astype(str) if desc_col else ""

        ctx_df = ctx_df.dropna(subset=["Start"])

        # Preview
        st.markdown("### Context Data Preview")
        ctx_df = st.data_editor(
            ctx_df,
            num_rows="dynamic",
            use_container_width=True
        )


        # ======= CHART: CT + event overlays =======
        st.markdown("### Cycle Time with Context Overlays")

        if selected_team == "All Teams":
            st.info("Please select a specific Team at the top.")
        else:
            team_ctx = ctx_df[ctx_df["Team"] == selected_team]

            if team_ctx.empty:
                st.info(f"No context events found for {selected_team}.")
            else:
                # Build monthly CT series using your existing column names
                if "End" in view_df.columns and "CT" in view_df.columns:
                    temp = view_df.dropna(subset=["End", "CT"]).copy()
                    temp["Month"] = pd.to_datetime(temp["End"]).dt.to_period("M").dt.to_timestamp()

                    monthly = temp.groupby("Month", as_index=False).agg(
                        median_ct=("CT", "median"),
                        p85_ct=("CT", lambda x: x.quantile(0.85)),
                        count=("CT", "count")
                    )

                    base = alt.Chart(monthly).encode(
                        x=alt.X("Month:T", title="Month")
                    )

                    ct_line = base.mark_line(point=True).encode(
                        y=alt.Y("median_ct:Q", title="Median CT (days)"),
                        tooltip=["Month:T", "median_ct:Q", "count:Q"]
                    )

                    event_marks = alt.Chart(team_ctx).mark_rule(
                        strokeDash=[4, 4], size=2
                    ).encode(
                        x="Start:T",
                        color="EventType:N",
                        tooltip=["EventType:N", "Who:N", "Description:N", "Start:T", "End:T"]
                    )

                    st.altair_chart(ct_line + event_marks, use_container_width=True)

                # ===== TIMELINE LIST =====
                st.markdown("### Context Timeline")
                for _, r in team_ctx.sort_values("Start").iterrows():
                    st.markdown(
                        f"**{r['Start'].date()} — {r['EventType']}** "
                        f"{'('+r['Who']+')' if r['Who'] else ''}  \n"
                        f"{r['Description']}"
                    )

        # ======= HEATMAP =======
        st.markdown("### Context Load Heatmap (All Teams)")

        ctx_df["Month"] = ctx_df["Start"].dt.to_period("M").dt.to_timestamp()
        heat = ctx_df.groupby(["Team", "Month"]).size().reset_index(name="Events")

        heatmap = alt.Chart(heat).mark_rect().encode(
            x="Month:T",
            y="Team:N",
            color="Events:Q",
            tooltip=["Team", "Month:T", "Events"]
        )

        st.altair_chart(heatmap, use_container_width=True)
