import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import io
import base64
import textwrap
import math
import json

st.set_page_config(page_title="Cycle Time Analysis", layout="wide")

# ---------- SIDEBAR CONFIG ----------
st.sidebar.title("Cycle Time Analysis – Timepiece (OBSS)")

st.sidebar.markdown(
    """
Use this app to explore Cycle Time, slowest items and forecasting
based on Timepiece exports.

**How to use:**
1. Export a CSV from Timepiece (OBSS) with at least:
   - Issue key
   - Status, transitions
   - Created, Resolved / Done
   - Any custom fields you care about (e.g. Team, Type, Epic, Story Points)
2. Upload it in the main section.
3. Choose filters and explore.
"""
)

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Timepiece CSV",
    type=["csv"],
    help="Export from the OBSS Timepiece report with all the fields you need.",
)


# ---------- DATA LOADING ----------
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    # Normalise column names (so we can reference them safely)
    df.columns = [c.strip() for c in df.columns]
    return df


df: pd.DataFrame | None = None
if uploaded_file is not None:
    try:
        df = load_data(uploaded_file.getvalue())
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

if df is None:
    st.info("Upload a Timepiece CSV to get started.")
    st.stop()


# ---------- BASIC CLEANUP / DERIVED FIELDS ----------
def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce")


if "Created" in df.columns:
    df["Created"] = safe_to_datetime(df["Created"])
if "Resolved" in df.columns:
    df["Resolved"] = safe_to_datetime(df["Resolved"])
if "End" in df.columns:
    df["End"] = safe_to_datetime(df["End"])

# Derive CT in days
if "Created" in df.columns:
    end_col = "End" if "End" in df.columns else ("Resolved" if "Resolved" in df.columns else None)
    if end_col:
        df["CT_days"] = (df[end_col] - df["Created"]).dt.total_seconds() / (3600 * 24)
    else:
        df["CT_days"] = np.nan
else:
    df["CT_days"] = np.nan

# Fallback team column
team_col_candidates = [c for c in df.columns if c.lower() in ("team", "squad", "board", "project")]
team_col = team_col_candidates[0] if team_col_candidates else None
if team_col is None:
    df["Team"] = "All"
    team_col = "Team"

# Issue Type + Sprint friendly names
issue_type_col = next((c for c in df.columns if c.lower().replace(" ", "") in ("issuetype", "type")), None)
sprint_col = next((c for c in df.columns if c.lower() == "sprint"), None)


# ---------- GLOBAL FILTERS ----------
teams = sorted(df[team_col].dropna().unique().tolist())
selected_team = st.sidebar.selectbox("Team", ["All Teams"] + teams)

date_col = "End" if "End" in df.columns else ("Resolved" if "Resolved" in df.columns else None)
date_min, date_max = None, None
if date_col:
    valid_dates = df[date_col].dropna()
    if not valid_dates.empty:
        date_min, date_max = valid_dates.min(), valid_dates.max()

if date_min is not None and date_max is not None:
    start_date, end_date = st.sidebar.date_input(
        "Date range",
        value=(date_min.date(), date_max.date()),
        min_value=date_min.date(),
        max_value=date_max.date(),
    )
else:
    start_date, end_date = None, None

# Base filtered view
view_df = df.copy()

if selected_team != "All Teams":
    view_df = view_df[view_df[team_col] == selected_team]

if date_col and start_date and end_date:
    mask = (view_df[date_col] >= pd.to_datetime(start_date)) & (
        view_df[date_col] <= pd.to_datetime(end_date) + pd.Timedelta(days=1)
    )
    view_df = view_df[mask]

# Optional IssueType & Sprint filters if present
if issue_type_col and issue_type_col in view_df.columns:
    all_types = sorted(view_df[issue_type_col].dropna().unique().tolist())
    selected_types = st.sidebar.multiselect("Issue Types", all_types, default=all_types)
    if selected_types:
        view_df = view_df[view_df[issue_type_col].isin(selected_types)]

if sprint_col and sprint_col in view_df.columns:
    # Sprints can be multi-valued, so we only do a coarse filter here:
    all_sprints = (
        pd.Series(view_df[sprint_col].fillna(""))
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
    )
    all_sprints = sorted([s for s in all_sprints.unique().tolist() if s])
    selected_sprints = st.sidebar.multiselect("Sprint (coarse filter)", all_sprints)
    if selected_sprints:
        mask = view_df[sprint_col].fillna("").astype(str)
        # Coarse contains check
        mask = mask.apply(
            lambda x: any(sel in x for sel in selected_sprints)
        )
        view_df = view_df[mask]


# ---------- TAB LAYOUT ----------
tabs = st.tabs(["Cycle Time", "Slowest Items", "Forecasting", "Data", "Sprint Analysis", "Context"])


# ---------- OVERVIEW ----------
with tabs[0]:

    # ---- Merged Overview & Insights (view-only) ----
    st.subheader("Cycle Time — Overview & Insights" + ("" if selected_team == "All Teams" else f" — {selected_team}"))

    if view_df.empty:
        st.info("No rows match the current filters.")
    else:
        # Core stats
        ct_series = view_df["CT_days"].dropna()
        if ct_series.empty:
            st.info("No completed items with CT data in current filter.")
        else:
            avg = ct_series.mean()
            median = ct_series.median()
            p85 = ct_series.quantile(0.85)
            p95 = ct_series.quantile(0.95)
            count = len(ct_series)

            col_a, col_b, col_c, col_d, col_e = st.columns(5)
            col_a.metric("Count", f"{count}")
            col_b.metric("Average CT (days)", f"{avg:.2f}")
            col_c.metric("Median CT (days)", f"{median:.2f}")
            col_d.metric("P85 CT (days)", f"{p85:.2f}")
            col_e.metric("P95 CT (days)", f"{p95:.2f}")

            # Distribution chart with Fibonacci-ish overlay
            hist_source = pd.DataFrame({"CT_days": ct_series})
            base = alt.Chart(hist_source).transform_bin(
                "binCT", field="CT_days", bin={"step": 1}
            ).mark_bar(opacity=0.7).encode(
                x=alt.X("binCT:Q", bin="binned", title="CT (days)"),
                x2="binCT_end:Q",
                y=alt.Y("count()", title="Items"),
            )

            # Fibonacci-like markers: 1, 2, 3, 5, 8, 13...
            fibs = [1, 2, 3, 5, 8, 13, 21]
            fibs = [f for f in fibs if f <= max(1, ct_series.max())]
            fib_df = pd.DataFrame({"fib": fibs})
            fib_layer = alt.Chart(fib_df).mark_rule(color="red", strokeDash=[4, 4]).encode(
                x="fib:Q",
                tooltip=["fib:Q"],
            )

            st.altair_chart((base + fib_layer).properties(height=320), use_container_width=True)

            # Time-series by completion date if we have date_col
            if date_col:
                ts_df = view_df[~view_df[date_col].isna()].copy()
                if not ts_df.empty:
                    ts_df["Month"] = ts_df[date_col].dt.to_period("M").dt.to_timestamp()

                    # Monthly stats
                    group = ts_df.groupby("Month")["CT_days"]
                    ts_stats = group.agg(
                        count="count",
                        avg="mean",
                        median="median",
                        p85=lambda x: x.quantile(0.85),
                    ).reset_index()

                    line = alt.Chart(ts_stats).mark_line(point=True).encode(
                        x=alt.X("Month:T", title="Month"),
                        y=alt.Y("median:Q", title="Median CT (days)"),
                        tooltip=[
                            alt.Tooltip("Month:T", title="Month"),
                            alt.Tooltip("count:Q", title="# items"),
                            alt.Tooltip("avg:Q", title="Avg CT", format=".2f"),
                            alt.Tooltip("median:Q", title="Median CT", format=".2f"),
                            alt.Tooltip("p85:Q", title="P85 CT", format=".2f"),
                        ],
                        color=alt.value("#1f77b4"),
                    )

                    band = alt.Chart(ts_stats).mark_area(opacity=0.15).encode(
                        x="Month:T",
                        y="median:Q",
                        y2="p85:Q",
                    )

                    st.altair_chart((band + line).properties(height=320), use_container_width=True)

            # Quick textual insights
            st.markdown("### Quick Insights")

            bullet_points = []
            bullet_points.append(
                f"- **Median CT** is **{median:.1f} days**, with P85 at **{p85:.1f} days**."
            )

            if p85 > 2 * median:
                bullet_points.append(
                    "- The tail (P85) is more than twice the median. There are some long-running outliers dragging the tail."
                )
            else:
                bullet_points.append(
                    "- The tail (P85) is reasonably close to the median, indicating a tighter distribution."
                )

            # You can add more rule-based insights here, e.g. based on type/priority/size distributions

            st.markdown("\n".join(bullet_points))


# ---------- SLOWEST ITEMS ----------
with tabs[1]:
    st.subheader("Slowest Items")

    if view_df.empty:
        st.info("No data in current filter set.")
    else:
        top_n = st.slider("How many items to show?", 5, 100, 20)
        slowest = view_df.sort_values("CT_days", ascending=False).head(top_n)

        key_col = next((c for c in slowest.columns if c.lower().replace(" ", "") in ("issuekey", "key")), None)
        summary_col = next((c for c in slowest.columns if c.lower() in ("summary", "title", "description")), None)

        display_cols = []
        if key_col:
            display_cols.append(key_col)
        if summary_col:
            display_cols.append(summary_col)
        for col in [team_col, issue_type_col, "CT_days", date_col]:
            if col and col not in display_cols and col in slowest.columns:
                display_cols.append(col)

        st.dataframe(slowest[display_cols], use_container_width=True)

        # Provide quick download of the slowest items
        csv = slowest.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download slowest items CSV",
            data=csv,
            file_name="slowest_items.csv",
            mime="text/csv",
        )


# ---------- FORECASTING ----------
with tabs[2]:
    st.subheader("Forecasting")

    if view_df.empty:
        st.info("No data in current filter set.")
    else:
        ct_values = view_df["CT_days"].dropna()
        if ct_values.empty:
            st.info("No CT values to forecast from.")
        else:
            st.markdown(
                """
                This section uses a simple Monte Carlo approach to project
                how long it might take to complete a batch of items,
                based purely on your historical CT distribution.
                """
            )

            batch_size = st.number_input(
                "Batch size (number of items)",
                min_value=1,
                max_value=500,
                value=10,
                step=1,
            )

            trials = st.number_input(
                "Number of simulations",
                min_value=200,
                max_value=5000,
                value=1000,
                step=100,
            )

            ct_array = ct_values.values

            @st.cache_data(show_spinner=False)
            def run_monte_carlo(ct_array, n_items, n_trials):
                rng = np.random.default_rng(seed=42)
                draws = rng.choice(ct_array, size=(n_trials, n_items), replace=True)
                totals = draws.sum(axis=1)
                return totals

            totals = run_monte_carlo(ct_array, int(batch_size), int(trials))

            p50 = np.percentile(totals, 50)
            p85 = np.percentile(totals, 85)
            p95 = np.percentile(totals, 95)

            col_f1, col_f2, col_f3 = st.columns(3)
            col_f1.metric("P50 (days)", f"{p50:.1f}")
            col_f2.metric("P85 (days)", f"{p85:.1f}")
            col_f3.metric("P95 (days)", f"{p95:.1f}")

            forecast_df = pd.DataFrame({"Total_CT_days": totals})
            chart = alt.Chart(forecast_df).transform_bin(
                "binCT", field="Total_CT_days", bin={"maxbins": 30}
            ).mark_bar().encode(
                x=alt.X("binCT:Q", bin="binned", title="Simulated total CT (days)"),
                x2="binCT_end:Q",
                y=alt.Y("count()", title="Count"),
            )

            st.altair_chart(chart.properties(height=320), use_container_width=True)


# ---------- RAW DATA VIEW ----------
with tabs[3]:
    st.subheader("Raw Data View")

    if view_df.empty:
        st.info("No data to show.")
    else:
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
                # Build a simple monthly series from view_df
                if date_col:
                    ts_df_ctx = view_df[~view_df[date_col].isna()].copy()
                    ts_df_ctx["month"] = ts_df_ctx[date_col].dt.to_period("M").dt.to_timestamp()

                    group_ctx = ts_df_ctx.groupby("month")["CT_days"]
                    monthly = group_ctx.agg(
                        median_ct="median",
                        count="count",
                    ).reset_index()
                else:
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
                            alt.Tooltip(
                                "median_ct:Q", title="Median CT", format=".2f"
                            ),
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
    # Index 4 == "Sprint Analysis" (we added "Context" as index 5)
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

            # Order sprints by end date median
            sprint_order = (
                expanded.groupby("Sprint")["_enddate"]
                .median()
                .sort_values()
                .index
                .tolist()
            )

            # 1) Sprint vs CT (median) line chart
            ct_by_sprint = (
                expanded.groupby("Sprint")["CT_days"]
                .agg(["count", "mean", "median", lambda x: x.quantile(0.85)])
                .reset_index()
            )
            ct_by_sprint.columns = ["Sprint", "Count", "AvgCT", "MedianCT", "P85CT"]
            ct_by_sprint["Sprint"] = pd.Categorical(ct_by_sprint["Sprint"], categories=sprint_order, ordered=True)
            ct_by_sprint = ct_by_sprint.sort_values("Sprint")

            sprint_chart = alt.Chart(ct_by_sprint).mark_line(point=True).encode(
                x=alt.X("Sprint:N", sort=sprint_order, title="Sprint"),
                y=alt.Y("MedianCT:Q", title="Median CT (days)"),
                tooltip=[
                    alt.Tooltip("Sprint:N", title="Sprint"),
                    alt.Tooltip("Count:Q", title="# Items"),
                    alt.Tooltip("AvgCT:Q", title="Avg CT", format=".2f"),
                    alt.Tooltip("MedianCT:Q", title="Median CT", format=".2f"),
                    alt.Tooltip("P85CT:Q", title="P85 CT", format=".2f"),
                ],
            )

            if sprint_tab is not None:
                with sprint_tab:
                    st.markdown("### CT by Sprint (Median with P85)")
                    st.altair_chart(sprint_chart.properties(height=320), use_container_width=True)

            # 2) Sprint / Type composition (stacked bar)
            comp = (
                expanded.groupby(["Sprint", "IssueType"])
                .size()
                .reset_index(name="Count")
            )
            comp["Sprint"] = pd.Categorical(comp["Sprint"], categories=sprint_order, ordered=True)
            comp = comp.sort_values("Sprint")

            comp_chart = alt.Chart(comp).mark_bar().encode(
                x=alt.X("Sprint:N", sort=sprint_order, title="Sprint"),
                y=alt.Y("Count:Q", stack="normalize", title="Proportion"),
                color=alt.Color("IssueType:N", title="Issue Type"),
                tooltip=[
                    alt.Tooltip("Sprint:N", title="Sprint"),
                    alt.Tooltip("IssueType:N", title="Issue Type"),
                    alt.Tooltip("Count:Q", title="Count"),
                ],
            )

            if sprint_tab is not None:
                with sprint_tab:
                    st.markdown("### Sprint Composition by Issue Type")
                    st.altair_chart(comp_chart.properties(height=320), use_container_width=True)
