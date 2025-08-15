
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
    st.caption("Upload data • Map columns • Team rules • Filters")

    # Upload
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
    st.caption("Used only if a Team column is not selected.")
    default_rules_text = "Team 1 = ^C7SM-\\d+\\b\nTeam 2 = ^C7O-\\d+\\b\nTeam 3 = ^C7F-\\d+\\b\nTeam 4 = ^C7T4-\\d+\\b"
    rules_text = st.text_area("Rules (Team = regex, one per line)", value=default_rules_text, height=120)

    st.markdown("### Cycle time mode")
    ct_mode = st.radio(
        "How to calculate CT",
        options=["Business days (Mon–Fri)", "Calendar days (elapsed)"],
        index=0,
        help=(
            "Business days: excludes weekends (Mon–Fri). Good for process improvement.\n"
            "Calendar days: total elapsed days, counting both start and finish days. Good for customer/SLA view."
        )
    )

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
    s = series.copy()
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
    ct_business = pd.Series(pd.to_numeric(ct_business, errors="coerce"), index=out.index).mask(ct_business < 0)
    ct_calendar = pd.to_numeric(ct_calendar, errors="coerce").mask(ct_calendar < 0)
    out["CT_business"] = ct_business
    out["CT_calendar"] = ct_calendar
    out["_date_issue"] = date_issue
    return out

def weekly_throughput(df: pd.DataFrame, started_col: str, finished_col: str) -> pd.Series:
    tmp = df.copy()
    base = auto_to_datetime(tmp.get("Finished", pd.Series(index=tmp.index))).fillna(auto_to_datetime(tmp[started_col]))
    week = base.dt.to_period("W-MON").dt.start_time
    s = week.value_counts().sort_index()
    s.name = "throughput"
    return s

def monthly_rollup(df: pd.DataFrame, started_col: str, finished_col: str) -> pd.DataFrame:
    tmp = df.copy()
    base = auto_to_datetime(tmp.get("Finished", pd.Series(index=tmp.index))).fillna(auto_to_datetime(tmp[started_col]))
    tmp["Month"] = pd.to_datetime(base.dt.to_period("M").dt.to_timestamp(), errors="coerce")
    g_all = tmp.groupby(["Team","Month"], dropna=False)
    items = g_all.size().rename("items")
    stats = tmp.groupby(["Team","Month"])["CT"].agg(avg="mean", p85=lambda s: s.quantile(0.85))
    out = items.to_frame().join(stats, how="left").reset_index()
    return out

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

# ===================== DATA PREP =====================
if st.session_state["data"].empty:
    st.info("Upload data to get started.")
    st.stop()

# Compute CTs and Teams first
raw = compute_ct_both(st.session_state["data"], started_col, finished_col)
raw = apply_team_mapping(raw, built_in_team_col, key_col, rules_text)

# Apply CT mode
raw["CT"] = raw["CT_business"] if ct_mode.startswith("Business") else raw["CT_calendar"]
ct_label = "Business days (Mon–Fri)" if ct_mode.startswith("Business") else "Calendar days (elapsed)"

# ===================== MAIN PAGE: TEAM SELECTOR & DATE RANGE =====================
st.title("Cycle Time Analysis")

# Build team list from the prepared data (no regex filtering here)
all_teams = sorted([t for t in raw["Team"].dropna().unique().tolist() if t != "Unassigned"])
if "Unassigned" in raw["Team"].unique():
    all_teams += ["Unassigned"]
team_options = ["All Teams"] + all_teams

selected_team = st.selectbox("Select team", team_options, index=0, help="Applies to all charts and tables on this page.")

# Filter view by selected team
if selected_team == "All Teams":
    view_df = raw.copy()
else:
    view_df = raw[raw["Team"] == selected_team].copy()

# ===================== TABS =====================
tabs = st.tabs(["Overview", "Teams", "Forecast", "Data"])

# ---------- OVERVIEW ----------
with tabs[0]:
    st.subheader("Overview" + ("" if selected_team == "All Teams" else f" — {selected_team}"))
    col1, col2, col3, col4 = st.columns(4)
    avg_ct = round(view_df["CT"].mean(), 2) if view_df["CT"].notna().any() else np.nan
    p85_ct = round(view_df["CT"].quantile(0.85), 2) if view_df["CT"].notna().any() else np.nan
    weekly = weekly_throughput(view_df, started_col, finished_col)
    thr = int(weekly[-4:].mean()) if len(weekly) >= 1 else 0
    items_this_month = 0
    roll = monthly_rollup(view_df, started_col, finished_col)
    if not roll.empty:
        latest_month = roll["Month"].max()
        items_this_month = int(roll.loc[roll["Month"]==latest_month, "items"].sum())

    for c, label, val in [(col1,"Average CT",avg_ct),
                          (col2,"85th CT",p85_ct),
                          (col3,"Throughput (wk avg)",thr),
                          (col4,"Items this month",items_this_month)]:
        c.metric(label, "--" if (isinstance(val,float) and np.isnan(val)) else val)

    base = alt.Chart(roll.dropna(subset=["Month"])).encode(x=alt.X("Month:T", title="Month", sort="ascending"))
    st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("avg:Q", title="Average CT")), use_container_width=True)
    st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("p85:Q", title="85th CT")), use_container_width=True)
    st.altair_chart(base.mark_bar().encode(y=alt.Y("items:Q", title="Item Count")), use_container_width=True)

    issues = int(view_df["_date_issue"].sum())
    if issues > 0:
        st.warning(f"{issues} rows have date parsing/ordering issues (start/finish). They were excluded from CT.")

# ---------- TEAMS ----------
with tabs[1]:
    st.subheader("Team deep-dive" + ("" if selected_team == "All Teams" else f" — {selected_team}"))
    finished = auto_to_datetime(view_df.get("Finished", pd.Series(index=view_df.index)))
    scatter_df = pd.DataFrame({"Finished": finished, "CT": view_df["CT"]}).dropna()
    if not scatter_df.empty:
        scatter = alt.Chart(scatter_df).mark_circle(size=50, opacity=0.7).encode(
            x=alt.X("Finished:T", title="Finish date"),
            y=alt.Y("CT:Q", title="Cycle time"),
            tooltip=["Finished","CT"]
        )
        st.altair_chart(scatter, use_container_width=True)

    roll = monthly_rollup(view_df, started_col, finished_col)
    base = alt.Chart(roll.dropna(subset=["Month"])).encode(x=alt.X("Month:T", title="Month", sort="ascending"))
    st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("avg:Q", title="Average CT")), use_container_width=True)
    st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("p85:Q", title="85th CT")), use_container_width=True)
    st.altair_chart(base.mark_bar().encode(y=alt.Y("items:Q", title="Item Count")), use_container_width=True)

# ---------- FORECAST ----------
with tabs[2]:
    st.subheader("Monte Carlo Forecast (Weekly)")
    if selected_team == "All Teams":
        st.info("Select a single team to run a per-team forecast.")
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

                        # Patched: pre-aggregate counts to avoid empty bars
                        df_sums = pd.DataFrame({"items": sums.astype(int)})
                        counts = (
                            df_sums.value_counts("items")
                            .sort_index()
                            .rename_axis("items")
                            .reset_index(name="sims")
                        )
                        hist = alt.Chart(counts).mark_bar().encode(
                            x=alt.X("items:Q", title="Items completed"),
                            y=alt.Y("sims:Q", title="Simulations"),
                            tooltip=["items","sims"]
                        )
                        st.altair_chart(hist, use_container_width=True)

                        # Cumulative probability
                        cum = counts.assign(cum=lambda d: d["sims"].cumsum(),
                                            prob=lambda d: d["sims"].cumsum() / d["sims"].sum())
                        cum_chart = alt.Chart(cum).mark_line(point=True).encode(
                            x=alt.X("items:Q", title="Items completed"),
                            y=alt.Y("prob:Q", title="Cumulative probability", axis=alt.Axis(format="%"))
                        )
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

                            # Patched: work with integer weeks; flag censored
                            cap = 1000
                            weeks_needed = weeks_needed.astype(int)
                            censored = int((weeks_needed >= cap).sum())
                            if censored > 0:
                                st.warning(f"{censored} simulations didn’t finish within {cap} weeks and were excluded from the chart.")

                            wk = weeks_needed[weeks_needed < cap]
                            wk_counts = (
                                pd.Series(wk, name="weeks")
                                  .value_counts()
                                  .sort_index()
                                  .rename_axis("weeks")
                                  .reset_index(name="sims")
                            )

                            wk_hist = alt.Chart(wk_counts).mark_bar().encode(
                                x=alt.X("weeks:Q", title="Weeks to finish (simulated)"),
                                y=alt.Y("sims:Q", title="Simulations"),
                                tooltip=["weeks","sims"]
                            )
                            st.altair_chart(wk_hist, use_container_width=True)

                            # Also show as dates without temporal binning
                            today = pd.Timestamp.today().normalize()
                            next_monday = (today + pd.offsets.Week(weekday=0))
                            date_counts = wk_counts.copy()
                            date_counts["date"] = (next_monday + pd.to_timedelta(date_counts["weeks"], unit="W")).dt.date
                            date_hist = alt.Chart(date_counts).mark_bar().encode(
                                x=alt.X("date:T", title="Finish date"),
                                y=alt.Y("sims:Q", title="Simulations"),
                                tooltip=["date","sims"]
                            )
                            st.altair_chart(date_hist, use_container_width=True)

                            txt = []
                            for c in sorted(conf_levels):
                                q = np.percentile(weeks_needed, c)
                                date_q = (next_monday + pd.to_timedelta(int(np.ceil(q)), unit="W")).date()
                                txt.append(f"{c}% → **{date_q}**")
                            st.markdown("**Forecast summary:** " + "  •  ".join(txt))

# ---------- DATA ----------
with tabs[3]:
    st.subheader("Data preview")
    st.caption("Rows flagged with date issues are excluded from CT calculations.")
    st.dataframe(view_df.assign(DateIssue=view_df["_date_issue"]).head(500))
