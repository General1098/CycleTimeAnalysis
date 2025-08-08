
import streamlit as st
import pandas as pd
import io
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import altair as alt
import numpy as np

st.set_page_config(page_title="Cycle Time Tracker", layout="wide")

st.title("Cycle Time Tracker")

# ========== 1) UPLOAD ==========
with st.expander("1) Upload data or start from scratch", expanded=True):
    uploaded = st.file_uploader("Upload JIRA export (CSV or Excel)", type=["csv", "xlsx"])
    if "data" not in st.session_state:
        st.session_state["data"] = pd.DataFrame()
    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = None
                for enc in ("utf-8", "utf-8-sig", "cp1252"):
                    try:
                        uploaded.seek(0)
                        df = pd.read_csv(uploaded, encoding=enc)
                        break
                    except Exception:
                        continue
                if df is None:
                    raise RuntimeError("CSV failed to load with encodings utf-8, utf-8-sig, cp1252")
            else:
                df = pd.read_excel(uploaded)
            st.session_state["data"] = df
            st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        except Exception as e:
            st.error(f"Failed to load: {e}")
    st.write("Current data shape:", st.session_state["data"].shape)
    if not st.session_state["data"].empty:
        st.dataframe(st.session_state["data"].head(50))

# ========== 2) COLUMN MAPPING ==========
with st.expander("2) Column mapping", expanded=True):
    df = st.session_state["data"]
    cols = list(df.columns) if not df.empty else []
    started_col = st.selectbox("Started column", options=["(none)"] + cols, index=(cols.index("Started")+1 if "Started" in cols else 0))
    finished_col = st.selectbox("Finished column", options=["(none)"] + cols, index=(cols.index("Finished")+1 if "Finished" in cols else 0))
    key_col = st.selectbox("Issue Key column", options=["(none)"] + cols, index=(cols.index("Key")+1 if "Key" in cols else 0))

# ========== 3) CYCLE TIME SETTINGS ==========
with st.expander("3) Cycle time settings", expanded=True):
    mode = st.radio("How should cycle time be calculated?", ["Calendar days", "Working days (Mon–Fri)"], index=1)
    decimals = st.number_input("Round cycle time to N decimals", min_value=0, max_value=4, value=1, step=1)
    dayfirst = st.checkbox("Parse dates as day-first (DD/MM/YYYY)", value=False)

def to_datetime_safe(series: pd.Series, dayfirst: bool=False) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, utc=False)

# ========== 4) TEAM MAPPING ==========
with st.expander("4) Team mapping", expanded=True):
    st.caption("If your dataset already has a 'Team' column, select it. Otherwise define rules to infer teams from the Issue Key.")
    cols = list(st.session_state["data"].columns) if not st.session_state["data"].empty else []
    built_in_team_col = st.selectbox("Use an existing Team column (optional)", options=["(none)"] + cols)
    rules_help = """
    **Rules syntax:** One rule per line as `Team Name = regex`.\n
    Example:\n
    Team 1 = ^C7SM-\\d+\\b\n
    Team 2 = ^C7F-\\d+\\b
    """
    st.markdown(rules_help)
    default_rules = r"Team 1 = ^C7SM-\\d+\\b\nTeam 2 = ^C7F-\\d+\\b"
    rules_text = st.text_area("Team rules (if no Team column)", value=default_rules, height=120)

def working_days_diff(s: pd.Series, f: pd.Series, dayfirst: bool=False) -> pd.Series:
    s = to_datetime_safe(s, dayfirst)
    f = to_datetime_safe(f, dayfirst)
    mask = s.notna() & f.notna()
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    if mask.any():
        start_d = s[mask].dt.floor("D").to_numpy(dtype="datetime64[D]")
        finish_d = f[mask].dt.floor("D").to_numpy(dtype="datetime64[D]")
        busdays = np.busday_count(start_d, finish_d)
        delta_days = (f[mask] - s[mask]).dt.total_seconds() / 86400.0
        cal_days = (finish_d - start_d).astype("timedelta64[D]").astype(int)
        cal_days = np.where(cal_days == 0, 1, cal_days)
        ratio = busdays / cal_days
        out.loc[mask] = delta_days * ratio
    return out

def calendar_days_diff(s: pd.Series, f: pd.Series, dayfirst: bool=False) -> pd.Series:
    s = to_datetime_safe(s, dayfirst)
    f = to_datetime_safe(f, dayfirst)
    return (f - s).dt.total_seconds() / 86400.0

def compute_ct(df: pd.DataFrame, started_col: str, finished_col: str, mode: str, decimals: int, dayfirst: bool) -> pd.DataFrame:
    out = df.copy()
    if started_col == "(none)" or finished_col == "(none)":
        st.warning("Please map both Started and Finished columns.")
        return out
    if mode.startswith("Working"):
        out["CT"] = working_days_diff(out[started_col], out[finished_col], dayfirst).round(decimals)
    else:
        out["CT"] = calendar_days_diff(out[started_col], out[finished_col], dayfirst).round(decimals)
    out["CT"] = pd.to_numeric(out["CT"], errors="coerce")
    return out

def parse_rules(txt: str) -> List[Dict[str, str]]:
    rules = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            team, pattern = line.split("=", 1)
            rules.append({"team": team.strip(), "pattern": pattern.strip()})
    return rules

def apply_team_mapping(df: pd.DataFrame, built_in_col: Optional[str], key_col: str, rules_text: str) -> pd.DataFrame:
    out = df.copy()
    if built_in_col and built_in_col != "(none)" and built_in_col in out.columns:
        out["Team"] = out[built_in_col].astype(str)
        return out
    rules = parse_rules(rules_text)
    def infer_team(key: str) -> str:
        if pd.isna(key):
            return "Unassigned"
        k = str(key)
        for r in rules:
            if re.search(r["pattern"], k):
                return r["team"]
        return "Unassigned"
    if key_col and key_col != "(none)" and key_col in out.columns:
        out["Team"] = out[key_col].apply(infer_team)
    else:
        out["Team"] = "Unassigned"
    return out

def monthly_rollup(df: pd.DataFrame, started_col: str, finished_col: str, dayfirst: bool=False) -> pd.DataFrame:
    tmp = df.copy()
    finished = to_datetime_safe(tmp.get("Finished", pd.Series(index=tmp.index)), dayfirst)
    if finished.notna().sum() == 0:
        base = to_datetime_safe(tmp[started_col], dayfirst)
    else:
        base = finished.fillna(to_datetime_safe(tmp[started_col], dayfirst))
    tmp["Month"] = pd.to_datetime(base.dt.to_period("M").dt.to_timestamp(), errors="coerce")
    g = tmp.groupby(["Team", "Month"], dropna=False)["CT"]
    out = g.agg(
        count="count",
        avg=lambda s: pd.to_numeric(s, errors="coerce").mean(),
        p85=lambda s: pd.to_numeric(s, errors="coerce").quantile(0.85)
    ).reset_index()
    return out

def weekly_throughput(df: pd.DataFrame, started_col: str, finished_col: str, dayfirst: bool=False) -> pd.Series:
    """Returns a series indexed by week-start of how many items finished that week."""
    tmp = df.copy()
    finished = to_datetime_safe(tmp.get("Finished", pd.Series(index=tmp.index)), dayfirst)
    base = finished.fillna(to_datetime_safe(tmp[started_col], dayfirst))
    # Use Monday as week start
    week = base.dt.to_period("W-MON").dt.start_time
    s = week.value_counts().sort_index()
    s.name = "throughput"
    return s

# ========== 5) COMPUTE + DASHBOARD + FORECAST TABS ==========
df = st.session_state["data"]
if df.empty:
    st.info("Upload data first.")
else:
    result = compute_ct(df, started_col, finished_col, mode, decimals, dayfirst)
    result = apply_team_mapping(result, built_in_team_col, key_col, rules_text)

    tabs = st.tabs(["Dashboard", "Monte Carlo Forecast"])

    # ----- Dashboard Tab -----
    with tabs[0]:
        with st.expander("Per-item results (with Team & CT)"):
            st.dataframe(result)

        teams = sorted(result["Team"].dropna().unique())
        if not teams:
            teams = ["Unassigned"]
        sel_team = st.selectbox("Choose a team", options=teams, key="dash_team")

        team_df = result[result["Team"] == sel_team].copy()
        if team_df.empty:
            st.warning(f"No rows for team '{sel_team}'. Check your mapping or upload.")
        else:
            roll = monthly_rollup(team_df, started_col, finished_col, dayfirst)

            with st.expander("Diagnostics — monthly rollup (preview)"):
                st.write(roll.head(20))
                st.write("dtypes:", roll.dtypes)

            base = alt.Chart(roll.dropna(subset=["Month"])).encode(
                x=alt.X("Month:T", title="Month", sort="ascending")
            )

            chart_avg = base.mark_line(point=True).encode(
                y=alt.Y("avg:Q", title="Average CT (days)")
            )
            if roll["avg"].notna().sum() > 2:
                trend_avg = base.transform_regression("Month", "avg").mark_line(strokeDash=[4, 4])
                chart_avg = chart_avg + trend_avg
            st.subheader(f"{sel_team} — Average CT")
            st.altair_chart(chart_avg, use_container_width=True)

            chart_p85 = base.mark_line(point=True).encode(
                y=alt.Y("p85:Q", title="85th CT (days)")
            )
            if roll["p85"].notna().sum() > 2:
                trend_p85 = base.transform_regression("Month", "p85").mark_line(strokeDash=[4, 4])
                chart_p85 = chart_p85 + trend_p85
            st.subheader(f"{sel_team} — 85th CT")
            st.altair_chart(chart_p85, use_container_width=True)

            chart_count = base.mark_bar().encode(
                y=alt.Y("count:Q", title="Item Count")
            )
            st.subheader(f"{sel_team} — Item Count")
            st.altair_chart(chart_count, use_container_width=True)

    # ----- Monte Carlo Forecast Tab -----
    with tabs[1]:
        st.subheader("Monte Carlo Forecast (Weekly)")
        teams = sorted(result["Team"].dropna().unique())
        sel_team_mc = st.selectbox("Team", options=teams, key="mc_team")

        # Controls
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            scenario = st.selectbox("Scenario", ["Items by date", "Date for items"])
        with col2:
            lookback_weeks = st.number_input("Use last N weeks", min_value=4, max_value=104, value=26, step=1)
        with col3:
            sims = st.number_input("Simulations", min_value=1000, max_value=50000, value=10000, step=1000)
        with col4:
            conf_levels = st.multiselect("Confidence levels", [50, 85, 95], default=[50, 85, 95])

        team_df_mc = result[result["Team"] == sel_team_mc].copy()
        if team_df_mc.empty:
            st.warning("No data for this team.")
        else:
            # Throughput vector
            weekly = weekly_throughput(team_df_mc, started_col, finished_col, dayfirst).sort_index()
            # Apply lookback
            cutoff = weekly.index.max() - pd.Timedelta(weeks=int(lookback_weeks))
            weekly_lb = weekly[weekly.index > cutoff]
            if len(weekly_lb) < 4:
                st.warning("Not enough weekly throughput history in the selected window. Increase 'Use last N weeks'.")
            else:
                st.write(f"Weekly throughput samples used: {len(weekly_lb)} (min={int(weekly_lb.min())}, median={int(weekly_lb.median())}, max={int(weekly_lb.max())})")

                if scenario == "Items by date":
                    target_date = st.date_input("Target date")
                    if target_date:
                        # Weeks between now (start of next week) and target_date
                        today = pd.Timestamp.today().normalize()
                        # Start from next Monday
                        next_monday = (today + pd.offsets.Week(weekday=0))
                        n_weeks = max(1, ((pd.to_datetime(target_date) - next_monday).days // 7) + 1)
                        samples = weekly_lb.to_numpy()
                        # Simulate sums
                        sums = np.random.choice(samples, size=(int(sims), int(n_weeks)), replace=True).sum(axis=1)
                        # Charts
                        df_sums = pd.DataFrame({"items": sums})
                        hist = alt.Chart(df_sums).mark_bar().encode(
                            x=alt.X("items:Q", bin=alt.Bin(maxbins=30), title="Items completed"),
                            y=alt.Y("count():Q", title="Frequency")
                        )
                        st.altair_chart(hist, use_container_width=True)
                        # Cumulative
                        cum = df_sums["items"].value_counts().sort_index().cumsum()
                        cum = (cum / cum.max()).reset_index()
                        cum.columns = ["items", "prob"]
                        cum_chart = alt.Chart(cum).mark_line(point=True).encode(
                            x="items:Q", y=alt.Y("prob:Q", axis=alt.Axis(format="%"), title="Cumulative probability")
                        )
                        st.altair_chart(cum_chart, use_container_width=True)
                        # Summary
                        txt = []
                        for c in sorted(conf_levels):
                            q = np.percentile(sums, c)
                            txt.append(f"{c}% → **{int(np.floor(q))}** items")
                        st.markdown("**Forecast summary:** " + "  •  ".join(txt))

                else:  # Date for items
                    backlog = st.number_input("How many items do we need to finish?", min_value=1, value=40, step=1)
                    if backlog:
                        samples = weekly_lb.to_numpy()
                        # Simulate time to finish: sample weeks until cumulative sum >= backlog
                        def sim_weeks_to_finish(target: int) -> int:
                            total = 0
                            weeks = 0
                            # Guard against rare zero-throughput weeks: ensure at least 1 item/week min=0 allowed; if all zeros, bail
                            if samples.max() == 0:
                                return np.iinfo(np.int32).max
                            while total < target and weeks < 1000:
                                total += np.random.choice(samples)
                                weeks += 1
                            return weeks
                        weeks_needed = np.array([sim_weeks_to_finish(int(backlog)) for _ in range(int(sims))])
                        # Convert to dates from next Monday
                        today = pd.Timestamp.today().normalize()
                        next_monday = (today + pd.offsets.Week(weekday=0))
                        dates = next_monday + pd.to_timedelta(weeks_needed, unit="W")
                        df_dates = pd.DataFrame({"date": dates})
                        # Histogram by week
                        hist = alt.Chart(df_dates).mark_bar().encode(
                            x=alt.X("date:T", bin=alt.Bin(maxbins=30), title="Finish date"),
                            y=alt.Y("count():Q", title="Frequency")
                        )
                        st.altair_chart(hist, use_container_width=True)
                        # Cumulative
                        cum = df_dates["date"].value_counts().sort_index().cumsum()
                        cum = (cum / cum.max()).reset_index()
                        cum.columns = ["date", "prob"]
                        cum_chart = alt.Chart(cum).mark_line(point=True).encode(
                            x=alt.X("date:T", title="Finish date"),
                            y=alt.Y("prob:Q", axis=alt.Axis(format="%"), title="Cumulative probability")
                        )
                        st.altair_chart(cum_chart, use_container_width=True)
                        # Summary
                        txt = []
                        for c in sorted(conf_levels):
                            q = np.percentile(weeks_needed, c)
                            date_q = (next_monday + pd.to_timedelta(int(np.ceil(q)), unit="W")).date()
                            txt.append(f"{c}% → **{date_q}**")
                        st.markdown("**Forecast summary:** " + "  •  ".join(txt))
