
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
    st.caption("Upload data • Map columns • Team rules • Settings")

    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
    if "data" not in st.session_state:
        st.session_state["data"] = pd.DataFrame()

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_try = None
                for enc in ("utf-8","utf-8-sig","cp1252"):
                    try:
                        uploaded.seek(0)
                        df_try = pd.read_csv(uploaded, encoding=enc)
                        break
                    except Exception:
                        continue
                if df_try is None:
                    raise RuntimeError("Could not read CSV with utf-8 / utf-8-sig / cp1252 encodings.")
                st.session_state["data"] = df_try
            else:
                st.session_state["data"] = pd.read_excel(uploaded)
            st.success(f"Loaded {len(st.session_state['data'])} rows")
        except Exception as e:
            st.error(f"Failed to load: {e}")

    df = st.session_state["data"]
    cols = list(df.columns) if not df.empty else []

    st.markdown("### Mapping")
    started_col = st.selectbox("Started column", options=["(none)"] + cols, index=(cols.index("Started")+1 if "Started" in cols else 0))
    finished_col = st.selectbox("Finished column", options=["(none)"] + cols, index=(cols.index("Finished")+1 if "Finished" in cols else 0))
    key_col = st.selectbox("Issue Key column", options=["(none)"] + cols, index=(cols.index("Key")+1 if "Key" in cols else 0))
    summary_col = st.selectbox("Summary/Title column (optional)", options=["(none)"] + cols, index=(cols.index("Summary")+1 if "Summary" in cols else 0))
    built_in_team_col = st.selectbox("Team column (optional)", options=["(none)"] + cols)

    st.markdown("### Team rules")
    st.caption("Used only if a Team column is not selected.")
    default_rules = "Team 1 = ^C7SM-\\d+\\b\nTeam 2 = ^C7O-\\d+\\b\nTeam 3 = ^C7F-\\d+\\b\nTeam 4 = ^C7T4-\\d+\\b"
    rules_text = st.text_area("Rules (Team = regex, one per line)", value=default_rules, height=120)

    st.markdown("### Cycle time mode")
    ct_mode = st.radio(
        "CT mode (when using dates)",
        options=["Business days (Mon–Fri)", "Calendar days (elapsed)"],
        index=0,
        help=(
            "Business days: excludes weekends (Mon–Fri).\n"
            "Calendar days: total elapsed days, counting both start and finish days."
        )
    )

    st.markdown("### Cycle time source")
    ct_source = st.radio(
        "Source",
        options=["Dates (Started/Finished)", "Duration text column(s)"],
        index=0,
        help="Use standard dates or sum one/more text duration columns like '0d 2h 36m 50s'."
    )

    duration_cols = []
    round_mode = "None"
    bucket_date_col = "(none)"
    if ct_source == "Duration text column(s)":
        # suggest likely text cols
        str_cols = [c for c in cols if df[c].dtype == "object"]
        duration_cols = st.multiselect(
            "Duration columns to sum (e.g., 'Dev Time', 'Wait Time')",
            options=str_cols,
            help="Values like '0d 2h 36m 50s'. You can pick multiple; they will be added together."
        )
        round_mode = st.selectbox(
            "Rounding",
            options=["None", "Ceil to hours", "Ceil to days"],
            help="Round the combined duration up to the nearest hour or day."
        )
        # Choose a date for bucketing charts (month/week). Prefer Finished, then Started.
        bucket_candidates = ["(none)"] + cols
        default_bucket_idx = 0
        if "Finished" in cols:
            default_bucket_idx = cols.index("Finished") + 1
        elif "Started" in cols:
            default_bucket_idx = cols.index("Started") + 1
        bucket_date_col = st.selectbox(
            "Date column for bucketing charts (month/week)",
            options=bucket_candidates,
            index=default_bucket_idx,
            help="Used to place items on a timeline when using duration columns."
        )

# ===================== HELPERS =====================
def parse_rules(txt: str) -> List[Dict[str,str]]:
    rules = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
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

def parse_duration_text(val) -> pd.Timedelta:
    """Parse strings like '0d 2h 36m 50s', '2h 5m', '30m', '45s' into Timedelta. Returns NaT on failure."""
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, (int, float, np.number)):
        # assume seconds if numeric
        try:
            return pd.to_timedelta(float(val), unit="s")
        except Exception:
            return pd.NaT
    s = str(val).strip().lower()
    if not s:
        return pd.NaT
    parts = re.findall(r'(\d+(?:\.\d+)?)\s*([dhms])', s)
    if not parts:
        return pd.NaT
    total_seconds = 0.0
    for num, unit in parts:
        v = float(num)
        if unit == "d":
            total_seconds += v * 86400
        elif unit == "h":
            total_seconds += v * 3600
        elif unit == "m":
            total_seconds += v * 60
        elif unit == "s":
            total_seconds += v
    return pd.to_timedelta(total_seconds, unit="s")

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

    out["CT_business"] = pd.Series(pd.to_numeric(ct_business, errors="coerce"), index=out.index).mask(ct_business < 0)
    out["CT_calendar"] = pd.to_numeric(ct_calendar, errors="coerce").mask(ct_calendar < 0)
    out["_date_issue"] = date_issue
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

def weekly_throughput(df: pd.DataFrame, started_col: str, finished_col: str) -> pd.Series:
    tmp = df.copy()
    if "_bucketer" in tmp.columns:
        base = pd.to_datetime(tmp["_bucketer"], errors="coerce")
    else:
        base = auto_to_datetime(tmp.get("Finished", pd.Series(index=tmp.index))).fillna(auto_to_datetime(tmp.get(started_col, pd.Series(index=tmp.index))))
    week = base.dt.to_period("W-MON").dt.start_time
    s = week.value_counts().sort_index()
    s.name = "throughput"
    return s

def monthly_rollup(df: pd.DataFrame, started_col: str, finished_col: str) -> pd.DataFrame:
    tmp = df.copy()
    if "_bucketer" in tmp.columns:
        base = pd.to_datetime(tmp["_bucketer"], errors="coerce")
    else:
        base = auto_to_datetime(tmp.get("Finished", pd.Series(index=tmp.index))).fillna(auto_to_datetime(tmp.get(started_col, pd.Series(index=tmp.index))))
    tmp["Month"] = pd.to_datetime(base.dt.to_period("M").dt.to_timestamp(), errors="coerce")

    g_team = tmp.groupby(["Team","Month"], dropna=False)
    items_team = g_team.size().rename("items")
    stats_team = tmp.groupby(["Team","Month"])["CT"].agg(avg="mean", p85=lambda s: s.quantile(0.85))
    per_team = items_team.to_frame().join(stats_team, how="left").reset_index()

    g_all = tmp.groupby("Month")
    items_all = g_all.size().rename("items").to_frame()
    stats_all = tmp.groupby("Month")["CT"].agg(avg="mean", p85=lambda s: s.quantile(0.85))
    all_df = items_all.join(stats_all, how="left").reset_index()
    all_df["Team"] = "All Teams"
    all_df = all_df[["Team","Month","items","avg","p85"]]

    out = pd.concat([per_team, all_df], ignore_index=True)
    return out

def slowest_items(view_df: pd.DataFrame, started_col: str, finished_col: str, team: str) -> pd.DataFrame:
    tmp = view_df.copy()
    if "_bucketer" in tmp.columns:
        base = pd.to_datetime(tmp["_bucketer"], errors="coerce")
    else:
        base = auto_to_datetime(tmp.get("Finished", pd.Series(index=tmp.index))).fillna(auto_to_datetime(tmp.get(started_col, pd.Series(index=tmp.index))))
    tmp["Month"] = pd.to_datetime(base.dt.to_period("M").dt.to_timestamp(), errors="coerce")

    if team != "All Teams":
        tmp = tmp[tmp["Team"] == team]

    th = tmp.groupby("Month")["CT"].quantile(0.85).rename("p85").reset_index()
    merged = tmp.merge(th, on="Month", how="left")
    out = merged[(merged["CT"].notna()) & (merged["CT"] > merged["p85"])].copy()
    return out

# ===================== DATA PREP =====================
if st.session_state["data"].empty:
    st.info("Upload data to get started.")
    st.stop()

if ct_source == "Dates (Started/Finished)":
    raw = compute_ct_both(st.session_state["data"], started_col, finished_col)
    raw = apply_team_mapping(raw, built_in_team_col, key_col, rules_text)
    raw["CT"] = raw["CT_business"] if ct_mode.startswith("Business") else raw["CT_calendar"]
    ct_label = "Business days (Mon–Fri)" if ct_mode.startswith("Business") else "Calendar days (elapsed)"
    bucketer = auto_to_datetime(raw.get("Finished", pd.Series(index=raw.index))).fillna(auto_to_datetime(raw.get(started_col, pd.Series(index=raw.index))))
else:
    raw = st.session_state["data"].copy()
    if len(duration_cols) == 0:
        st.error("Please select at least one duration column.")
        st.stop()
    tot = pd.Series(0.0, index=raw.index)
    any_parsed = pd.Series(False, index=raw.index)
    for c in duration_cols:
        td = raw[c].apply(parse_duration_text)
        secs = td.dt.total_seconds()
        any_parsed = any_parsed | secs.notna()
        secs = secs.fillna(0.0)
        tot = tot + secs
    if round_mode == "Ceil to hours":
        tot = np.ceil(tot / 3600.0) * 3600.0
    elif round_mode == "Ceil to days":
        tot = np.ceil(tot / 86400.0) * 86400.0
    raw["CT"] = (tot / 86400.0).astype(float)
    raw["_date_issue"] = ~any_parsed
    raw = apply_team_mapping(raw, built_in_team_col, key_col, rules_text)
    raw["CT_business"] = np.nan
    raw["CT_calendar"] = np.nan
    ct_label = f"Duration columns ({' + '.join(duration_cols)})"
    if bucket_date_col and bucket_date_col != "(none)" and bucket_date_col in raw.columns:
        bucketer = auto_to_datetime(raw[bucket_date_col])
    else:
        bucketer = pd.Series(pd.NaT, index=raw.index)

# Attach bucketer so weekly/monthly functions can use it
raw["_bucketer"] = bucketer

# ===================== MAIN PAGE: TEAM SELECTOR =====================
st.title("Cycle Time Analysis")

teams = sorted([t for t in raw["Team"].dropna().unique().tolist() if t != "Unassigned"])
if "Unassigned" in raw["Team"].unique():
    teams += ["Unassigned"]
team_options = ["All Teams"] + teams
selected_team = st.selectbox("Select Team", team_options, index=0, help="Applies across all tabs below.")

# Filter the view
if selected_team == "All Teams":
    view_df = raw.copy()
else:
    view_df = raw[raw["Team"] == selected_team].copy()

st.caption(f"CT source: **{('Dates' if ct_source.startswith('Dates') else 'Durations')}** • Using **{ct_label}** for CT.")

# ===================== TABS =====================
tabs = st.tabs(["Overview", "Cycle Time", "Slowest Items", "WIP & Work Item Age", "Forecast", "Data"])

# ---------- OVERVIEW ----------
with tabs[0]:
    st.subheader("Overview" + ("" if selected_team == "All Teams" else f" — {selected_team}"))
    col1, col2, col3, col4 = st.columns(4)

    avg_ct = round(view_df["CT"].mean(), 2) if view_df["CT"].notna().any() else np.nan
    p85_ct = round(view_df["CT"].quantile(0.85), 2) if view_df["CT"].notna().any() else np.nan
    weekly = weekly_throughput(view_df, started_col, finished_col)
    thr = int(weekly[-4:].mean()) if len(weekly) >= 1 else 0

    roll = monthly_rollup(view_df, started_col, finished_col)
    if selected_team == "All Teams":
        roll_sel = roll[roll["Team"] == "All Teams"].copy()
    else:
        roll_sel = roll[roll["Team"] == selected_team].copy()

    items_this_month = 0
    if not roll_sel.empty:
        latest_month = roll_sel["Month"].max()
        items_this_month = int(roll_sel.loc[roll_sel["Month"] == latest_month, "items"].sum())

    col1.metric("Average CT", "--" if (isinstance(avg_ct,float) and np.isnan(avg_ct)) else avg_ct)
    col2.metric("85th CT", "--" if (isinstance(p85_ct,float) and np.isnan(p85_ct)) else p85_ct)
    col3.metric("Throughput (wk avg)", thr)
    col4.metric("Items this month", items_this_month)

# ---------- CYCLE TIME ----------
with tabs[1]:
    st.subheader("Cycle Time" + ("" if selected_team == "All Teams" else f" — {selected_team}"))
    if not roll_sel.empty:
        base = alt.Chart(roll_sel.dropna(subset=["Month"])).encode(x=alt.X("Month:T", title="Month", sort="ascending"))
        st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("avg:Q", title="Average CT")), use_container_width=True)
        st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("p85:Q", title="85th CT")), use_container_width=True)
        st.altair_chart(base.mark_bar().encode(y=alt.Y("items:Q", title="Item Count")), use_container_width=True)

    st.markdown("**CT Distribution (last 6 months)**")
    if "_bucketer" in view_df.columns:
        base_dt = pd.to_datetime(view_df["_bucketer"], errors="coerce")
    else:
        base_dt = auto_to_datetime(view_df.get("Finished", pd.Series(index=view_df.index))).fillna(auto_to_datetime(view_df.get(started_col, pd.Series(index=view_df.index))))
    month_cut = (base_dt.max() - pd.offsets.DateOffset(months=6)) if pd.notna(base_dt.max()) else None
    if month_cut is not None:
        mask = base_dt >= pd.Timestamp(month_cut)
        dist_df = pd.DataFrame({"CT": view_df.loc[mask, "CT"]}).dropna()
        if not dist_df.empty:
            hist = alt.Chart(dist_df).mark_bar().encode(
                x=alt.X("CT:Q", bin=alt.Bin(maxbins=30), title="CT (days)"),
                y=alt.Y("count()", title="Items")
            )
            st.altair_chart(hist, use_container_width=True)
        else:
            st.info("No CT data in the last 6 months for the selected scope.")

# ---------- SLOWEST ITEMS ----------
with tabs[2]:
    st.subheader("Slowest Items" + ("" if selected_team == "All Teams" else f" — {selected_team}"))
    si = slowest_items(view_df, started_col, finished_col, "All Teams" if selected_team == "All Teams" else selected_team)
    if si.empty:
        st.info("No items above the 85th percentile threshold for the selected scope.")
    else:
        show_cols = []
        if key_col != "(none)" and key_col in si.columns: show_cols.append(key_col)
        if summary_col != "(none)" and summary_col in si.columns: show_cols.append(summary_col)
        show_cols += ["Team","CT","Month"]
        st.dataframe(si[show_cols].sort_values(["Month","CT"], ascending=[False, False]).head(200), use_container_width=True)

        count_slow = si.groupby("Month").size().rename("slow_items").reset_index()
        chart = alt.Chart(count_slow).mark_bar().encode(
            x=alt.X("Month:T", title="Month"),
            y=alt.Y("slow_items:Q", title="Items above 85th CT")
        )
        st.altair_chart(chart, use_container_width=True)

# ---------- WIP & WORK ITEM AGE ----------
with tabs[3]:
    st.subheader("WIP & Work Item Age" + ("" if selected_team == "All Teams" else f" — {selected_team}"))
    started = auto_to_datetime(view_df.get(started_col, pd.Series(index=view_df.index))) if started_col != "(none)" else pd.Series(pd.NaT, index=view_df.index)
    finished = auto_to_datetime(view_df.get(finished_col, pd.Series(index=view_df.index))) if finished_col != "(none)" else pd.Series(pd.NaT, index=view_df.index)

    open_mask = started.notna() & finished.isna()
    wip_df = view_df[open_mask].copy()
    today = pd.Timestamp.today().normalize()
    wip_df["Age_days"] = (today - started[open_mask]).dt.days

    c1, c2 = st.columns(2)
    c1.metric("Current WIP", int(wip_df.shape[0]))
    if wip_df.shape[0] > 0:
        c2.metric("Avg Work Item Age (days)", int(wip_df["Age_days"].mean()))
        age_hist = alt.Chart(wip_df).mark_bar().encode(
            x=alt.X("Age_days:Q", bin=alt.Bin(maxbins=30), title="Age (days)"),
            y=alt.Y("count()", title="Items")
        )
        st.altair_chart(age_hist, use_container_width=True)
        cols_to_show = []
        if key_col != "(none)" and key_col in wip_df.columns: cols_to_show.append(key_col)
        if summary_col != "(none)" and summary_col in wip_df.columns: cols_to_show.append(summary_col)
        cols_to_show += ["Team"]
        st.dataframe(wip_df[cols_to_show + ["Age_days"]].sort_values("Age_days", ascending=False).head(300), use_container_width=True)
    else:
        st.info("No in-progress items found (Started set, Finished empty).")

# ---------- FORECAST ----------
with tabs[4]:
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
                        today_ts = pd.Timestamp.today().normalize()
                        next_monday = (today_ts + pd.offsets.Week(weekday=0))
                        n_weeks = max(1, ((pd.to_datetime(target_date) - next_monday).days // 7) + 1)
                        sums = np.random.choice(samples, size=(int(sims), int(n_weeks)), replace=True).sum(axis=1)

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

                            today_ts = pd.Timestamp.today().normalize()
                            next_monday = (today_ts + pd.offsets.Week(weekday=0))
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
with tabs[5]:
    st.subheader("Data preview")
    st.caption("Rows flagged with date issues are excluded from CT calculations.")
    st.dataframe(view_df.assign(DateIssue=view_df["_date_issue"]).head(1000), use_container_width=True)
