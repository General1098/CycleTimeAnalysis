
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from typing import Optional, Dict, List

st.set_page_config(page_title="Cycle Time Analysis", layout="wide")

# ===================== SIDEBAR: SETTINGS & IMPORT =====================
with st.sidebar:
    st.title("Cycle Time Analysis")
    st.caption("Upload data • Map columns • Pick team")

    # Upload
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"], label_visibility="collapsed")
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
                    raise RuntimeError("Could not read CSV with utf-8 / utf-8-sig / cp1252 encodings.")
            else:
                df = pd.read_excel(uploaded)
            st.session_state["data"] = df
            st.success(f"Loaded {len(df)} rows")
        except Exception as e:
            st.error(f"Failed to load: {e}")

    df = st.session_state["data"]
    cols = list(df.columns) if not df.empty else []

    st.markdown("### Mapping")
    started_col = st.selectbox("Started", options=["(none)"] + cols, index=(cols.index("Started")+1 if "Started" in cols else 0))
    finished_col = st.selectbox("Finished", options=["(none)"] + cols, index=(cols.index("Finished")+1 if "Finished" in cols else 0))
    key_col = st.selectbox("Issue Key", options=["(none)"] + cols, index=(cols.index("Key")+1 if "Key" in cols else 0))
    built_in_team_col = st.selectbox("Team column (optional)", options=["(none)"] + cols)

    st.markdown("### Team rules")
    st.caption("Only used if no Team column is selected.")
    default_rules = "Team 1 = ^C7SM-\\d+\\b\nTeam 2 = ^C7O-\\d+\\b\nTeam 3 = ^C7F-\\d+\\b\nTeam 4 = ^C7T4-\\d+\\b"
    rules_text = st.text_area("Rules (one per line: Team = regex)", value=default_rules, height=120, label_visibility="collapsed")

# ===================== HELPERS =====================

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
    Robust mixed-format parser:
    - Handles both MM/DD/YYYY and DD/MM/YYYY in the same column.
    - Heuristic: if first token > 12 and second <= 12 => day-first for that row.
    - Falls back to pandas' parser otherwise.
    - Leaves unparseable values as NaT.
    """
    s = series.copy()

    # If already datetime-like, coerce and return
    try:
        return pd.to_datetime(s, errors="coerce", utc=False)
    except Exception:
        pass

    s = s.astype("object")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    is_str = s.apply(lambda x: isinstance(x, str))
    strs = s[is_str].str.strip()

    # Extract simple d/m/y or m/d/y
    m = strs.str.extract(r"^(?P<a>\\d{1,2})[\\/-](?P<b>\\d{1,2})[\\/-](?P<c>\\d{2,4})$", expand=True)
    simple_mask = m.notna().all(axis=1)
    if simple_mask.any():
        a = m.loc[simple_mask, "a"].astype(int)
        b = m.loc[simple_mask, "b"].astype(int)
        c = m.loc[simple_mask, "c"].astype(int)
        c = c.where(c > 99, c + 2000)  # two-digit years → 2000+

        # Decide per-row
        dayfirst_rows = (a > 12) & (b <= 12)
        monthfirst_rows = ~dayfirst_rows

        # Month-first
        if monthfirst_rows.any():
            mf = pd.to_datetime(pd.DataFrame({"year": c[monthfirst_rows],
                                              "month": a[monthfirst_rows],
                                              "day": b[monthfirst_rows]}), errors="coerce")
            out.loc[simple_mask[simple_mask].index[monthfirst_rows]] = mf.values
        # Day-first
        if dayfirst_rows.any():
            df_ = pd.to_datetime(pd.DataFrame({"year": c[dayfirst_rows],
                                               "month": b[dayfirst_rows],
                                               "day": a[dayfirst_rows]}), errors="coerce")
            out.loc[simple_mask[simple_mask].index[dayfirst_rows]] = df_.values

    # Remaining → let pandas try
    remaining = out.isna()
    if remaining.any():
        out.loc[remaining] = pd.to_datetime(s.loc[remaining], errors="coerce", utc=False)

    return out

def compute_ct(df: pd.DataFrame, started_col: str, finished_col: str, working_days: bool) -> pd.DataFrame:
    out = df.copy()
    if started_col == "(none)" or finished_col == "(none)":
        return out.assign(CT=np.nan, _date_issue=False)

    start = auto_to_datetime(out[started_col])
    finish = auto_to_datetime(out[finished_col])

    # Flag parsing/order issues
    date_issue = (start.isna() | finish.isna()) | (finish < start)

    if working_days:
        sD = start.dt.floor("D").to_numpy(dtype="datetime64[D]")
        fD = finish.dt.floor("D").to_numpy(dtype="datetime64[D]")
        # Using numpy busday_count; ignore invalids with masking later
        with np.errstate(invalid="ignore"):
            bus = np.busday_count(sD, fD).astype("float")
        delta_days = (finish - start).dt.total_seconds() / 86400.0
        cal_days = (fD - sD).astype("timedelta64[D]").astype("float")
        cal_days = np.where(cal_days == 0, 1.0, cal_days)
        ratio = np.divide(bus, cal_days, out=np.zeros_like(bus), where=cal_days!=0)
        ct = delta_days * ratio
    else:
        ct = (finish - start).dt.total_seconds() / 86400.0

    # Clean negatives / invalids
    ct = pd.to_numeric(ct, errors="coerce")
    ct = ct.mask(ct < 0)

    out["CT"] = ct
    out["_date_issue"] = date_issue
    return out

def monthly_rollup(df: pd.DataFrame, started_col: str, finished_col: str) -> pd.DataFrame:
    tmp = df.copy()
    base = auto_to_datetime(tmp.get("Finished", pd.Series(index=tmp.index)))
    base = base.fillna(auto_to_datetime(tmp[started_col]))
    tmp["Month"] = pd.to_datetime(base.dt.to_period("M").dt.to_timestamp(), errors="coerce")
    g = tmp.groupby(["Team","Month"], dropna=False)["CT"]
    return g.agg(count="count", avg="mean", p85=lambda s: s.quantile(0.85)).reset_index()

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

# ===================== BUILD RESULT =====================
working_days = True  # default business-days CT
if df.empty:
    st.info("Upload data to get started.")
    st.stop()

# Team rules preview
with st.sidebar:
    if key_col not in ("(none)", None) and key_col in df.columns:
        try:
            preview_rules = parse_rules(rules_text)
            keys = df[key_col].astype(str)
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

result = compute_ct(df, started_col, finished_col, working_days)
result = apply_team_mapping(result, built_in_team_col, key_col, rules_text)

# Team filter
teams = sorted(result["Team"].dropna().unique().tolist() or ["Unassigned"])
with st.sidebar:
    sel_team = st.selectbox("Team", options=teams, index=0)

# ===================== MAIN TABS =====================
tabs = st.tabs(["Overview", "Teams", "Forecast", "Data"])

# ---------- OVERVIEW ----------
with tabs[0]:
    st.subheader(f"Overview — {sel_team}")
    team_df = result[result["Team"] == sel_team].copy()

    col1, col2, col3, col4 = st.columns(4)
    avg_ct = round(team_df["CT"].mean(), 2) if team_df["CT"].notna().any() else np.nan
    p85_ct = round(team_df["CT"].quantile(0.85), 2) if team_df["CT"].notna().any() else np.nan
    weekly = weekly_throughput(team_df, started_col, finished_col)
    thr = int(weekly[-4:].mean()) if len(weekly) >= 1 else 0
    items_this_month = 0
    roll = monthly_rollup(team_df, started_col, finished_col)
    if not roll.empty:
        latest_month = roll["Month"].max()
        items_this_month = int(roll.loc[roll["Month"]==latest_month, "count"].sum())

    for c, label, val in [(col1,"Average CT (days)",avg_ct),
                          (col2,"85th CT (days)",p85_ct),
                          (col3,"Throughput (wk avg)",thr),
                          (col4,"Items this month",items_this_month)]:
        c.metric(label, "--" if (isinstance(val,float) and np.isnan(val)) else val)

    base = alt.Chart(roll.dropna(subset=["Month"])).encode(x=alt.X("Month:T", title="Month", sort="ascending"))
    st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("avg:Q", title="Average CT (days)")), use_container_width=True)
    st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("p85:Q", title="85th CT (days)")), use_container_width=True)
    st.altair_chart(base.mark_bar().encode(y=alt.Y("count:Q", title="Item Count")), use_container_width=True)

    issues = int(team_df["_date_issue"].sum())
    if issues > 0:
        st.warning(f"{issues} rows have date parsing or ordering issues (start/finish). They were excluded from CT averages.")

# ---------- TEAMS ----------
with tabs[1]:
    st.subheader("Team deep-dive")
    team_df = result[result["Team"] == sel_team].copy()
    finished = auto_to_datetime(team_df.get("Finished", pd.Series(index=team_df.index)))
    scatter_df = pd.DataFrame({"Finished": finished, "CT": team_df["CT"]}).dropna()
    if not scatter_df.empty:
        scatter = alt.Chart(scatter_df).mark_circle(size=50, opacity=0.7).encode(
            x=alt.X("Finished:T", title="Finish date"),
            y=alt.Y("CT:Q", title="Cycle time (days)"),
            tooltip=["Finished","CT"]
        )
        st.altair_chart(scatter, use_container_width=True)

    roll = monthly_rollup(team_df, started_col, finished_col)
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
        lookback_weeks = st.number_input("Use last N weeks", min_value=4, max_value=104, value=26, step=1)
    with colC:
        sims = st.number_input("Simulations", min_value=1000, max_value=50000, value=10000, step=1000)
    with colD:
        conf_levels = st.multiselect("Confidence levels", [50,85,95], default=[50,85,95])

    weekly = weekly_throughput(team_df, started_col, finished_col).sort_index()
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
    st.subheader("Data preview")
    st.caption("Rows flagged with date issues are excluded from CT averages.")
    st.dataframe(result.assign(DateIssue=result["_date_issue"]).head(200))
