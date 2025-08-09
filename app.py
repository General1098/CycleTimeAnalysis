
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from typing import Optional, Dict, List

# -------------------- PAGE & THEME --------------------
st.set_page_config(page_title="Cycle Time Analysis", layout="wide")

PRIMARY_RED = "#DC2626"  # brand red
DARK_BG = "#0b0f16"
DARK_SURFACE = "#111827"
LIGHT_TEXT = "#F9FAFB"
DARK_TEXT = "#0b0f16"

# Basic dark theme CSS (white/red branding)
st.markdown(f"""
<style>
:root {{
  --brand-red: {PRIMARY_RED};
}}
html, body, [data-testid="stAppViewContainer"] {{
  background: {DARK_BG};
  color: {LIGHT_TEXT};
}}
[data-testid="stSidebar"] {{
  background: #0a0e14;
  color: {LIGHT_TEXT};
}}
h1,h2,h3,h4,h5,h6,label,span,div,p,small,em,strong {{ color: {LIGHT_TEXT}; }}
.kpi {{
  border: 1px solid rgba(220,38,38,0.25);
  background: {DARK_SURFACE};
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 6px 24px rgba(0,0,0,0.35);
}}
.kpi .label {{ font-size: 12px; opacity: 0.85; }}
.kpi .value {{ font-size: 28px; font-weight: 800; color: {PRIMARY_RED}; }}
</style>
""", unsafe_allow_html=True)

# -------------------- HELPERS --------------------
def parse_rules(txt: str) -> List[Dict[str,str]]:
    rules = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            team, pattern = line.split("=", 1)
            rules.append({"team": team.strip(), "pattern": pattern.strip()})
    return rules

def auto_to_datetime(series: pd.Series) -> pd.Series:
    """
    Mixed-format parser: supports MM/DD/YYYY and DD/MM/YYYY per row.
    Heuristic: if first token > 12 and second <= 12 -> day-first.
    Falls back to pandas parser; invalid -> NaT.
    """
    s = series.copy()
    # if already datetime-like, just coerce
    try:
        return pd.to_datetime(s, errors="coerce", utc=False)
    except Exception:
        pass
    s = s.astype("object")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    is_str = s.apply(lambda x: isinstance(x, str))
    strs = s[is_str].str.strip()
    m = strs.str.extract(r"^(?P<a>\d{1,2})[\/-](?P<b>\d{1,2})[\/-](?P<c>\d{2,4})$", expand=True)
    simple_mask = m.notna().all(axis=1)
    if simple_mask.any():
        a = m.loc[simple_mask, "a"].astype(int)
        b = m.loc[simple_mask, "b"].astype(int)
        c = m.loc[simple_mask, "c"].astype(int)
        c = c.where(c > 99, c + 2000)
        dayfirst_rows = (a > 12) & (b <= 12)
        monthfirst_rows = ~dayfirst_rows
        if monthfirst_rows.any():
            mf = pd.to_datetime(pd.DataFrame({"year": c[monthfirst_rows],
                                              "month": a[monthfirst_rows],
                                              "day": b[monthfirst_rows]}), errors="coerce")
            out.loc[simple_mask[simple_mask].index[monthfirst_rows]] = mf.values
        if dayfirst_rows.any():
            df_ = pd.to_datetime(pd.DataFrame({"year": c[dayfirst_rows],
                                               "month": b[dayfirst_rows],
                                               "day": a[dayfirst_rows]}), errors="coerce")
            out.loc[simple_mask[simple_mask].index[dayfirst_rows]] = df_.values
    remaining = out.isna()
    if remaining.any():
        out.loc[remaining] = pd.to_datetime(s.loc[remaining], errors="coerce", utc=False)
    return out

def compute_ct_both(df: pd.DataFrame, started_col: str, finished_col: str) -> pd.DataFrame:
    out = df.copy()
    if started_col == "(none)" or finished_col == "(none)":
        out["CT_business"] = np.nan
        out["CT_calendar"] = np.nan
        out["_date_issue"] = False
        return out

    start = auto_to_datetime(out[started_col])
    finish = auto_to_datetime(out[finished_col])
    date_issue = (start.isna() | finish.isna()) | (finish < start)

    sD = start.dt.floor("D").to_numpy(dtype="datetime64[D]")
    fD = finish.dt.floor("D").to_numpy(dtype="datetime64[D]")
    with np.errstate(invalid="ignore"):
        ct_business = np.busday_count(sD, fD).astype("float")

    ct_calendar = (finish - start).dt.days + 1

    ct_business = pd.to_numeric(ct_business, errors="coerce")
    ct_business = pd.Series(ct_business, index=out.index).mask(ct_business < 0)
    ct_calendar = pd.to_numeric(ct_calendar, errors="coerce").mask(ct_calendar < 0)

    out["CT_business"] = ct_business
    out["CT_calendar"] = ct_calendar
    out["_date_issue"] = date_issue
    return out

def monthly_rollup(df: pd.DataFrame, started_col: str, finished_col: str) -> pd.DataFrame:
    tmp = df.copy()
    base = auto_to_datetime(tmp.get("Finished", pd.Series(index=tmp.index)))
    base = base.fillna(auto_to_datetime(tmp[started_col]))
    tmp["Month"] = pd.to_datetime(base.dt.to_period("M").dt.to_timestamp(), errors="coerce")
    g = tmp.groupby(["Team","Month"], dropna=False)["CT"]
    out = g.agg(count="count", avg="mean", p85=lambda s: s.quantile(0.85)).reset_index()
    return out

def weekly_throughput(df: pd.DataFrame, started_col: str, finished_col: str) -> pd.Series:
    tmp = df.copy()
    base = auto_to_datetime(tmp.get("Finished", pd.Series(index=tmp.index))).fillna(auto_to_datetime(tmp[started_col]))
    week = base.dt.to_period("W-MON").dt.start_time
    s = week.value_counts().sort_index()
    s.name = "throughput"
    return s

def apply_team_mapping(df: pd.DataFrame, built_in_team_col: Optional[str], key_col: str, rules_text: str) -> pd.DataFrame:
    out = df.copy()
    if built_in_team_col and built_in_team_col != "(none)" and built_in_team_col in out.columns:
        out["Team"] = out[built_in_team_col].astype(str)
        return out
    rules = parse_rules(rules_text)
    def infer_team(key):
        if pd.isna(key): return "Unassigned"
        k = str(key)
        for r in rules:
            try:
                if re.search(r["pattern"], k):
                    return r["team"]
            except re.error:
                continue
        return "Unassigned"
    if key_col and key_col != "(none)" and key_col in out.columns:
        out["Team"] = out[key_col].apply(infer_team)
    else:
        out["Team"] = "Unassigned"
    return out

# -------------------- SIDEBAR: DATA & SETTINGS --------------------
with st.sidebar:
    st.title("Upload & Map")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
    if "data" not in st.session_state:
        st.session_state["data"] = pd.DataFrame()

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = None
                for enc in ("utf-8","utf-8-sig","cp1252"):
                    try:
                        uploaded.seek(0)
                        df = pd.read_csv(uploaded, encoding=enc)
                        break
                    except Exception:
                        continue
                if df is None:
                    raise RuntimeError("Could not read CSV with utf-8/utf-8-sig/cp1252 encodings.")
            else:
                df = pd.read_excel(uploaded)
            st.session_state["data"] = df
            st.success(f"Loaded {len(df)} rows")
        except Exception as e:
            st.error(f"Failed to load: {e}")

    df = st.session_state["data"]
    cols = list(df.columns) if not df.empty else []

    st.subheader("Mapping")
    started_col = st.selectbox("Started", options=["(none)"] + cols, index=(cols.index("Started")+1 if "Started" in cols else 0))
    finished_col = st.selectbox("Finished", options=["(none)"] + cols, index=(cols.index("Finished")+1 if "Finished" in cols else 0))
    key_col = st.selectbox("Issue Key", options=["(none)"] + cols, index=(cols.index("Key")+1 if "Key" in cols else 0))
    built_in_team_col = st.selectbox("Team column (optional)", options=["(none)"] + cols)

    st.subheader("Team rules")
    st.caption("Only used if no Team column is selected.")
    default_rules = "Team 1 = ^C7SM-\\d+\\b\nTeam 2 = ^C7O-\\d+\\b\nTeam 3 = ^C7F-\\d+\\b\nTeam 4 = ^C7T4-\\d+\\b"
    rules_text = st.text_area("Rules (Team = regex, one per line)", value=default_rules, height=120)

    st.divider()
    ct_mode = st.radio(
        "Cycle time mode",
        options=["Business days (Mon–Fri)", "Calendar days (elapsed)"],
        index=0,
        help=(
            "Choose how cycle time is calculated:\n"
            "• Business days: excludes weekends (Mon–Fri). Good for process improvement.\n"
            "• Calendar days: total elapsed days, counting both start and finish days. Good for customer/SLA view."
        )
    )

# -------------------- DATA PREP --------------------
if st.session_state["data"].empty:
    st.info("Upload data to get started.")
    st.stop()

raw = compute_ct_both(st.session_state["data"], started_col, finished_col)
raw = apply_team_mapping(raw, built_in_team_col, key_col, rules_text)

# Apply CT mode
if ct_mode.startswith("Business"):
    raw["CT"] = raw["CT_business"]
    ct_label = "Business days (Mon–Fri)"
else:
    raw["CT"] = raw["CT_calendar"]
    ct_label = "Calendar days (elapsed)"

# Live preview of rule matches
with st.sidebar:
    if key_col not in ("(none)", None) and key_col in raw.columns:
        try:
            preview_rules = parse_rules(rules_text)
            keys = raw[key_col].astype(str)
            counts = []
            assigned = pd.Series(False, index=keys.index)
            for r in preview_rules:
                m = keys.str.contains(r["pattern"], regex=True, na=False)
                counts.append((r["team"], int(m.sum())))
                assigned |= m
            counts.append(("Unassigned", int((~assigned).sum())))
            st.caption("Rule match preview:")
            st.write(dict(counts))
        except re.error as e:
            st.error(f"Team rules have an invalid regex: {e}")

# -------------------- TOP BAR: TEAM SELECTOR --------------------
teams = ["All Teams"] + sorted([t for t in raw["Team"].dropna().unique().tolist() if t != "Unassigned"]) + (["Unassigned"] if "Unassigned" in raw["Team"].unique() else [])
st.markdown("### ")
top_col1, top_col2 = st.columns([2,3], vertical_alignment="center")
with top_col1:
    sel_team = st.selectbox("Team (applies to all charts)", options=teams, index=0)
with top_col2:
    st.caption(f"Using **{ct_label}**")

# Filter by team for charts
if sel_team == "All Teams":
    view_df = raw.copy()
else:
    view_df = raw[raw["Team"] == sel_team].copy()

# -------------------- TABS --------------------
tabs = st.tabs(["Overview", "Teams", "Forecast", "Data"])

# ---------- OVERVIEW ----------
with tabs[0]:
    st.subheader("Overview")
    # Monthly rollup (per selected view)
    def monthly_roll(df):
        tmp = df.copy()
        base = auto_to_datetime(tmp.get("Finished", pd.Series(index=tmp.index)))
        base = base.fillna(auto_to_datetime(tmp[started_col]))
        tmp["Month"] = pd.to_datetime(base.dt.to_period("M").dt.to_timestamp(), errors="coerce")
        g = tmp.groupby(["Team","Month"], dropna=False)["CT"]
        return g.agg(count="count", avg="mean", p85=lambda s: s.quantile(0.85)).reset_index()

    roll = monthly_roll(view_df)

    # KPI cards (if single team selected use that; if All Teams, combine)
    col1, col2, col3, col4 = st.columns(4)
    avg_ct = round(view_df["CT"].mean(), 2) if view_df["CT"].notna().any() else np.nan
    p85_ct = round(view_df["CT"].quantile(0.85), 2) if view_df["CT"].notna().any() else np.nan
    weekly = weekly_throughput(view_df, started_col, finished_col)
    thr = int(weekly[-4:].mean()) if len(weekly) >= 1 else 0
    items_this_month = 0
    if not roll.empty:
        latest_month = roll["Month"].max()
        items_this_month = int(roll.loc[roll["Month"]==latest_month, "count"].sum())

    for c, label, val in [(col1,"Average CT",avg_ct),
                          (col2,"85th CT",p85_ct),
                          (col3,"Throughput (wk avg)",thr),
                          (col4,"Items this month",items_this_month)]:
        c.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{("--" if (isinstance(val,float) and np.isnan(val)) else val)}</div></div>', unsafe_allow_html=True)

    # Charts (aggregate over selected view)
    base = alt.Chart(roll.dropna(subset=["Month"])).encode(
        x=alt.X("Month:T", title="Month", sort="ascending")
    )
    st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("avg:Q", title="Average CT")), use_container_width=True)
    st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("p85:Q", title="85th CT")), use_container_width=True)
    st.altair_chart(base.mark_bar().encode(y=alt.Y("count:Q", title="Item Count")), use_container_width=True)

    issues = int(view_df["_date_issue"].sum())
    if issues > 0:
        st.warning(f"{issues} rows have date parsing or ordering issues (start/finish). They were excluded from CT.")

# ---------- TEAMS (deep-dive for selected team or multi-team view) ----------
with tabs[1]:
    st.subheader("Team deep-dive")
    if sel_team == "All Teams":
        st.info("Select a specific team at the top to see item-level charts.")
    else:
        finished = auto_to_datetime(view_df.get("Finished", pd.Series(index=view_df.index)))
        scatter_df = pd.DataFrame({"Finished": finished, "CT": view_df["CT"]}).dropna()
        if not scatter_df.empty:
            scatter = alt.Chart(scatter_df).mark_circle(size=50, opacity=0.7).encode(
                x=alt.X("Finished:T", title="Finish date"),
                y=alt.Y("CT:Q", title="Cycle time"),
                tooltip=["Finished","CT"]
            )
            st.altair_chart(scatter, use_container_width=True)
        # Monthly roll for this team
        tmp_roll = monthly_roll(view_df)
        base = alt.Chart(tmp_roll.dropna(subset=["Month"])).encode(x=alt.X("Month:T", title="Month", sort="ascending"))
        st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("avg:Q", title="Average CT")), use_container_width=True)
        st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("p85:Q", title="85th CT")), use_container_width=True)
        st.altair_chart(base.mark_bar().encode(y=alt.Y("count:Q", title="Item Count")), use_container_width=True)

# ---------- FORECAST (per-team Monte Carlo) ----------
with tabs[2]:
    st.subheader("Monte Carlo Forecast (Weekly, per team)")
    if sel_team == "All Teams":
        st.info("Select a single team at the top to run per-team forecasts.")
    else:
        colA, colB, colC, colD = st.columns(4)
        with colA:
            scenario = st.selectbox("Scenario", ["Items by date", "Date for items"])
        with colB:
            lookback_weeks = st.number_input("Use last N weeks", min_value=4, max_value=104, value=26, step=1)
        with colC:
            sims = st.number_input("Simulations", min_value=1000, max_value=50000, value=10000, step=1000)
        with colD:
            conf_levels = st.multiselect("Confidence levels", [50,85,95], default=[50,85,95])

        weekly = weekly_throughput(view_df, started_col, finished_col).sort_index()
        if weekly.empty:
            st.info("No finished items yet — add data to run a forecast.")
        else:
            cutoff = weekly.index.max() - pd.Timedelta(weeks=int(lookback_weeks))
            weekly_lb = weekly[weekly.index > cutoff]
            if len(weekly_lb) < 4:
                st.warning("Not enough weekly history in the window. Increase 'Use last N weeks'.")
            else:
                st.caption(f"Samples used: {len(weekly_lb)} | min={int(weekly_lb.min())}, median={int(weekly_lb.median())}, max={int(weekly_lb.max())}")
                samples = weekly_lb.to_numpy()

                if scenario == "Items by date":
                    target_date = st.date_input("Target date")
                    if target_date:
                        today = pd.Timestamp.today().normalize()
                        next_monday = (today + pd.offsets.Week(weekday=0))
                        n_weeks = max(1, ((pd.to_datetime(target_date) - next_monday).days // 7) + 1)
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
                    backlog = st.number_input("How many items do we need to finish?", min_value=1, value=40, step=1)
                    if backlog:
                        if samples.max() == 0:
                            st.error("All sampled weeks are zero; cannot forecast.")
                        else:
                            def sim_weeks_to_finish(target: int) -> int:
                                total=0; weeks=0
                                while total < target and weeks < 1000:
                                    total += np.random.choice(samples)
                                    weeks += 1
                                return weeks
                            weeks_needed = np.array([sim_weeks_to_finish(int(backlog)) for _ in range(int(sims))])
                            today = pd.Timestamp.today().normalize()
                            next_monday = (today + pd.offsets.Week(weekday=0))
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
    st.subheader("Data")
    st.caption("Rows flagged with date issues are excluded from CT calculations.")
    st.dataframe(raw.assign(DateIssue=raw["_date_issue"]).head(500))
