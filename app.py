
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, List

# -------------------- THEME --------------------
st.set_page_config(page_title="Cycle Time Analysis", layout="wide")

# Sidebar theme toggle and global filters
with st.sidebar:
    st.title("⚙️ Settings")
    dark_mode = st.toggle("Dark mode", value=True)
    primary_red = "#DC2626"  # brand red
    bg_white = "#FFFFFF"
    bg_dark = "#0b0f16"
    surface_dark = "#111827"
    text_light = "#F9FAFB"
    text_dark = "#111827"
    # Inject CSS
    css = f"""
    <style>
    :root {{
      --accent: {primary_red};
      --bg: {'{bg_dark}'};
      --surface: {'{surface_dark}'};
      --text: {'{text_light}' if dark_mode else text_dark};
    }}
    html, body, [data-testid="stAppViewContainer"] {{
      background: {'#0b0f16' if dark_mode else '#FFFFFF'};
    }}
    [data-testid="stSidebar"] {{ background: {'#0a0e14' if dark_mode else '#FFFFFF'}; }}
    h1,h2,h3,h4,h5,h6, p, span, label, div {{ color: {'#F9FAFB' if dark_mode else '#0b0f16'} !important; }}
    /* KPI Cards */
    .kpi {{
      border: 1px solid rgba(220,38,38,0.25);
      background: {'#111827' if dark_mode else '#FFFFFF'};
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 4px 14px rgba(0,0,0,{0.4 if dark_mode else 0.06});
    }}
    .kpi .label {{ font-size: 12px; opacity: 0.8; }}
    .kpi .value {{ font-size: 28px; font-weight: 700; color: {primary_red}; }}
    .pill {{
      display:inline-block; padding:2px 8px; border-radius:999px;
      background: rgba(220,38,38,0.15); color: {primary_red}; font-size:12px; margin-left:8px;
    }}
    .section {{
      padding: 8px 0 0 0; border-top: 1px dashed rgba(148,163,184,0.2); margin-top: 8px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -------------------- HELPERS --------------------
def to_datetime_safe(series: pd.Series, dayfirst: bool=False) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, utc=False)

def working_days_diff(s: pd.Series, f: pd.Series, dayfirst: bool=False) -> pd.Series:
    s = to_datetime_safe(s, dayfirst); f = to_datetime_safe(f, dayfirst)
    mask = s.notna() & f.notna()
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    if mask.any():
        start_d = s[mask].dt.floor("D").to_numpy(dtype="datetime64[D]")
        finish_d = f[mask].dt.floor("D").to_numpy(dtype="datetime64[D]")
        busdays = np.busday_count(start_d, finish_d)
        delta_days = (f[mask] - s[mask]).dt.total_seconds() / 86400.0
        cal_days = (finish_d - start_d).astype("timedelta64[D]").astype(int)
        cal_days = np.where(cal_days == 0, 1, cal_days)
        ratio = np.divide(busdays, cal_days, out=np.zeros_like(busdays, dtype=float), where=cal_days!=0)
        out.loc[mask] = delta_days * ratio
    return out

def calendar_days_diff(s: pd.Series, f: pd.Series, dayfirst: bool=False) -> pd.Series:
    s = to_datetime_safe(s, dayfirst); f = to_datetime_safe(f, dayfirst)
    return (f - s).dt.total_seconds() / 86400.0

def compute_ct(df: pd.DataFrame, started_col: str, finished_col: str, mode: str, decimals: int, dayfirst: bool) -> pd.DataFrame:
    out = df.copy()
    if started_col == "(none)" or finished_col == "(none)":
        st.warning("Please map both Started and Finished columns.")
        return out
    out["CT"] = (working_days_diff(out[started_col], out[finished_col], dayfirst)
                 if mode.startswith("Working") else
                 calendar_days_diff(out[started_col], out[finished_col], dayfirst)).round(decimals)
    out["CT"] = pd.to_numeric(out["CT"], errors="coerce")
    return out

def parse_rules(txt: str) -> List[Dict[str, str]]:
    rules = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        if "=" in line:
            team, pattern = line.split("=", 1)
            rules.append({"team": team.strip(), "pattern": pattern.strip()})
    return rules

def apply_team_mapping(df: pd.DataFrame, built_in_col: Optional[str], key_col: str, rules_text: str) -> pd.DataFrame:
    out = df.copy()
    if built_in_col and built_in_col != "(none)" and built_in_col in out.columns:
        out["Team"] = out[built_in_team_col].astype(str); return out
    rules = parse_rules(rules_text)
    def infer_team(key: str) -> str:
        if pd.isna(key): return "Unassigned"
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
    base = finished.fillna(to_datetime_safe(tmp[started_col], dayfirst))
    tmp["Month"] = pd.to_datetime(base.dt.to_period("M").dt.to_timestamp(), errors="coerce")
    g = tmp.groupby(["Team", "Month"], dropna=False)["CT"]
    out = g.agg(count="count",
                avg=lambda s: pd.to_numeric(s, errors="coerce").mean(),
                p85=lambda s: pd.to_numeric(s, errors="coerce").quantile(0.85)).reset_index()
    return out

def weekly_throughput(df: pd.DataFrame, started_col: str, finished_col: str, dayfirst: bool=False) -> pd.Series:
    tmp = df.copy()
    finished = to_datetime_safe(tmp.get("Finished", pd.Series(index=tmp.index)), dayfirst)
    base = finished.fillna(to_datetime_safe(tmp[started_col], dayfirst))
    week = base.dt.to_period("W-MON").dt.start_time
    s = week.value_counts().sort_index()
    s.name = "throughput"
    return s

# -------------------- DATA LOAD & MAPPING --------------------
st.header("Cycle Time Analysis")
with st.expander("1) Upload data or start from scratch", expanded=True):
    uploaded = st.file_uploader("Upload JIRA export (CSV or Excel)", type=["csv","xlsx"])
    if "data" not in st.session_state:
        st.session_state["data"] = pd.DataFrame()
    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = None
                for enc in ("utf-8","utf-8-sig","cp1252"):
                    try:
                        uploaded.seek(0); df = pd.read_csv(uploaded, encoding=enc); break
                    except Exception: continue
                if df is None: raise RuntimeError("CSV failed to load with encodings utf-8, utf-8-sig, cp1252")
            else:
                df = pd.read_excel(uploaded)
            st.session_state["data"] = df
            st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        except Exception as e:
            st.error(f"Failed to load: {e}")
    if not st.session_state["data"].empty:
        st.dataframe(st.session_state["data"].head(30))

with st.expander("2) Column mapping", expanded=True):
    df = st.session_state["data"]
    cols = list(df.columns) if not df.empty else []
    started_col = st.selectbox("Started column", ["(none)"] + cols, index=(cols.index("Started")+1 if "Started" in cols else 0))
    finished_col = st.selectbox("Finished column", ["(none)"] + cols, index=(cols.index("Finished")+1 if "Finished" in cols else 0))
    key_col = st.selectbox("Issue Key column", ["(none)"] + cols, index=(cols.index("Key")+1 if "Key" in cols else 0))

with st.expander("3) Settings", expanded=True):
    mode = st.radio("Cycle time basis", ["Calendar days", "Working days (Mon–Fri)"], index=1, horizontal=True)
    decimals = st.number_input("Round cycle time to N decimals", 0, 4, 1)
    dayfirst = st.checkbox("Parse dates as day-first (DD/MM/YYYY)", value=False)

with st.expander("4) Team mapping", expanded=True):
    cols = list(st.session_state["data"].columns) if not st.session_state["data"].empty else []
    built_in_team_col = st.selectbox("Use an existing Team column (optional)", ["(none)"] + cols)
    st.caption("Or define rules (one per line): `Team Name = regex`")
    default_rules = r"Team 1 = ^C7SM-\d+\b\nTeam 2 = ^C7F-\d+\b"
    rules_text = st.text_area("Team rules", value=default_rules, height=120)

    # Live rule match counts (preview)
    if not st.session_state["data"].empty and key_col not in (None, "(none)"):
        try:
            rules = parse_rules(rules_text)
            key_series = st.session_state["data"][key_col].astype(str)
            counts = []
            assigned = pd.Series(False, index=key_series.index)
            for r in rules:
                match = key_series.str.contains(r["pattern"], regex=True, na=False)
                counts.append((r["team"], int(match.sum())))
                assigned |= match
            counts.append(("Unassigned", int((~assigned).sum())))
            st.write("**Rule match preview:**", dict(counts))
        except re.error as e:
            st.error(f"Invalid regex in team rules: {e}")

# -------------------- Derive Results --------------------
if st.session_state["data"].empty:
    st.info("Upload data to continue.")
    st.stop()

result = compute_ct(st.session_state["data"], started_col, finished_col, mode, decimals, dayfirst)
result = apply_team_mapping(result, built_in_team_col, key_col, rules_text)

# Sidebar filters
with st.sidebar:
    st.divider()
    st.subheader("Filters")
    teams = sorted(result["Team"].dropna().unique().tolist() or ["Unassigned"])
    sel_team = st.selectbox("Team", options=teams, index=0)

# -------------------- TABS --------------------
tabs = st.tabs(["Overview", "Teams", "Forecast", "Data", "Mapping"])

# ---------- OVERVIEW ----------
with tabs[0]:
    st.subheader(f"Overview — {sel_team}")
    team_df = result[result["Team"] == sel_team].copy()
    roll = monthly_rollup(team_df, started_col, finished_col, dayfirst)
    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    avg_ct = round(team_df["CT"].mean(), 2) if team_df["CT"].notna().any() else np.nan
    p85_ct = round(team_df["CT"].quantile(0.85), 2) if team_df["CT"].notna().any() else np.nan
    weekly = weekly_throughput(team_df, started_col, finished_col, dayfirst)
    thr = int(weekly[-4:].mean()) if len(weekly) >= 1 else 0
    for c, label, val in [(col1,"Average CT (days)",avg_ct),
                          (col2,"85th CT (days)",p85_ct),
                          (col3,"Throughput (wk avg)",thr),
                          (col4,"Items this month", int(roll[roll["Month"]==roll["Month"].max()]["count"].sum()) if len(roll)>0 else 0)]:
        with c:
            st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{val if not np.isnan(val) else "--"}</div></div>', unsafe_allow_html=True)

    base = alt.Chart(roll.dropna(subset=["Month"])).encode(x=alt.X("Month:T", title="Month", sort="ascending"))
    st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("avg:Q", title="Average CT (days)")), use_container_width=True)
    st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("p85:Q", title="85th CT (days)")), use_container_width=True)
    st.altair_chart(base.mark_bar().encode(y=alt.Y("count:Q", title="Item Count")), use_container_width=True)

# ---------- TEAMS ----------
with tabs[1]:
    st.subheader("Team Deep-dive")
    team_df = result[result["Team"] == sel_team].copy()
    if not team_df.empty:
        finished = to_datetime_safe(team_df.get("Finished", pd.Series(index=team_df.index)), dayfirst)
        scatter_df = pd.DataFrame({"Finished": finished, "CT": team_df["CT"]}).dropna()
        if not scatter_df.empty:
            scatter = alt.Chart(scatter_df).mark_circle(size=50, opacity=0.7).encode(
                x=alt.X("Finished:T", title="Finish date"),
                y=alt.Y("CT:Q", title="Cycle time (days)"),
                tooltip=["Finished","CT"]
            )
            st.altair_chart(scatter, use_container_width=True)
    roll = monthly_rollup(team_df, started_col, finished_col, dayfirst)
    base = alt.Chart(roll.dropna(subset=["Month"])).encode(x=alt.X("Month:T", title="Month", sort="ascending"))
    st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("avg:Q", title="Average CT (days)")), use_container_width=True)
    st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("p85:Q", title="85th CT (days)")), use_container_width=True)
    st.altair_chart(base.mark_bar().encode(y=alt.Y("count:Q", title="Item Count")), use_container_width=True)

# ---------- FORECAST ----------
with tabs[2]:
    st.subheader("Monte Carlo Forecast (Weekly)")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        scenario = st.selectbox("Scenario", ["Items by date", "Date for items"])
    with colB:
        lookback_weeks = st.number_input("Use last N weeks", 4, 104, 26, 1)
    with colC:
        sims = st.number_input("Simulations", 1000, 50000, 10000, 1000)
    with colD:
        conf_levels = st.multiselect("Confidence levels", [50,85,95], default=[50,85,95])
    weekly = weekly_throughput(team_df, started_col, finished_col, dayfirst).sort_index()
    cutoff = weekly.index.max() - pd.Timedelta(weeks=int(lookback_weeks)) if len(weekly)>0 else None
    weekly_lb = weekly[weekly.index > cutoff] if cutoff is not None else weekly
    if len(weekly_lb) < 4:
        st.warning("Not enough weekly throughput history in the selected window. Increase 'Use last N weeks'.")
    else:
        st.caption(f"Samples used: {len(weekly_lb)} | min={int(weekly_lb.min())}, median={int(weekly_lb.median())}, max={int(weekly_lb.max())}")
        if scenario == "Items by date":
            target_date = st.date_input("Target date")
            if target_date:
                today = pd.Timestamp.today().normalize()
                next_monday = (today + pd.offsets.Week(weekday=0))
                n_weeks = max(1, ((pd.to_datetime(target_date) - next_monday).days // 7) + 1)
                samples = weekly_lb.to_numpy()
                sums = np.random.choice(samples, size=(int(sims), int(n_weeks)), replace=True).sum(axis=1)
                df_sums = pd.DataFrame({"items": sums})
                hist = alt.Chart(df_sums).mark_bar().encode(x=alt.X("items:Q", bin=alt.Bin(maxbins=30), title="Items completed"), y="count()")
                st.altair_chart(hist, use_container_width=True)
                cum = df_sums["items"].value_counts().sort_index().cumsum()
                cum = (cum / cum.max()).reset_index(); cum.columns=["items","prob"]
                cum_chart = alt.Chart(cum).mark_line(point=True).encode(x="items:Q", y=alt.Y("prob:Q", axis=alt.Axis(format="%"), title="Cumulative probability"))
                st.altair_chart(cum_chart, use_container_width=True)
                txt = []
                for c in sorted(conf_levels):
                    q = np.percentile(sums, c); txt.append(f"{c}% → **{int(np.floor(q))}** items")
                st.markdown("**Forecast summary:** " + "  •  ".join(txt))
        else:
            backlog = st.number_input("How many items do we need to finish?", 1, 10000, 40, 1)
            if backlog:
                samples = weekly_lb.to_numpy()
                if samples.max() == 0:
                    st.error("All sampled weeks have zero throughput; cannot forecast.")
                else:
                    def sim_weeks_to_finish(target: int) -> int:
                        total=0; weeks=0
                        while total < target and weeks < 1000:
                            total += np.random.choice(samples); weeks += 1
                        return weeks
                    weeks_needed = np.array([sim_weeks_to_finish(int(backlog)) for _ in range(int(sims))])
                    today = pd.Timestamp.today().normalize(); next_monday = (today + pd.offsets.Week(weekday=0))
                    dates = next_monday + pd.to_timedelta(weeks_needed, unit="W")
                    df_dates = pd.DataFrame({"date": dates})
                    hist = alt.Chart(df_dates).mark_bar().encode(x=alt.X("date:T", bin=alt.Bin(maxbins=30), title="Finish date"), y="count()")
                    st.altair_chart(hist, use_container_width=True)
                    cum = df_dates["date"].value_counts().sort_index().cumsum()
                    cum = (cum / cum.max()).reset_index(); cum.columns=["date","prob"]
                    cum_chart = alt.Chart(cum).mark_line(point=True).encode(x=alt.X("date:T", title="Finish date"), y=alt.Y("prob:Q", axis=alt.Axis(format="%"), title="Cumulative probability"))
                    st.altair_chart(cum_chart, use_container_width=True)
                    txt = []
                    for c in sorted(conf_levels):
                        q = np.percentile(weeks_needed, c)
                        date_q = (next_monday + pd.to_timedelta(int(np.ceil(q)), unit="W")).date()
                        txt.append(f"{c}% → **{date_q}**")
                    st.markdown("**Forecast summary:** " + "  •  ".join(txt))

# ---------- DATA ----------
with tabs[3]:
    st.subheader("Data Quality & Uploads")
    started_na = result[started_col].isna().sum() if started_col in result.columns else 0
    finished_na = result[finished_col].isna().sum() if finished_col in result.columns else 0
    ct_na = result["CT"].isna().sum() if "CT" in result.columns else 0
    dup_keys = result.duplicated(subset=[key_col]).sum() if key_col in result.columns else 0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Missing Started", started_na)
    col2.metric("Missing Finished", finished_na)
    col3.metric("Missing CT", ct_na)
    col4.metric("Duplicate Keys", dup_keys)
    st.dataframe(result.head(200))

# ---------- MAPPING ----------
with tabs[4]:
    st.subheader("Mapping & Rules")
    st.write("**Current mapping**")
    st.json({"Started": started_col, "Finished": finished_col, "Key": key_col, "Team column": built_in_team_col})
    st.write("**Rules**")
    st.code(rules_text or "(none)")
