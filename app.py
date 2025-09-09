from __future__ import annotations
import io, json, math, re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dateutil import parser as dtparser

from timepiece_client import TimepieceClient

st.set_page_config(page_title="Cycle Time Tracker", layout="wide")
DEFAULT_WORKDAY_HOURS = 8

def parse_duration_text(text: str) -> Optional[timedelta]:
    if not isinstance(text, str):
        return None
    m = re.fullmatch(r"\s*(?:(?P<d>\d+)\s*d)?\s*(?:(?P<h>\d+)\s*h)?\s*(?:(?P<m>\d+)\s*m)?\s*(?:(?P<s>\d+)\s*s)?\s*", text.strip())
    if not m: return None
    return timedelta(days=int(m.group("d") or 0), hours=int(m.group("h") or 0), minutes=int(m.group("m") or 0), seconds=int(m.group("s") or 0))

def to_datetime_safe(x): 
    if pd.isna(x): return None
    if isinstance(x, datetime): return x
    try: return dtparser.parse(str(x))
    except: return None

def ceil_timedelta(td: timedelta, unit: str) -> timedelta:
    secs = td.total_seconds()
    if unit=="hours": return timedelta(hours=math.ceil(secs/3600))
    if unit=="days": return timedelta(days=math.ceil(secs/86400))
    return td

def business_hours_diff(start, end, workday_hours=DEFAULT_WORKDAY_HOURS):
    if not start or not end: return timedelta(0)
    if end<start: start,end=end,start
    bdays=np.busday_count(start.date(), end.date())
    if np.is_busday(end.date()): bdays+=1
    return timedelta(hours=bdays*workday_hours)

@dataclass
class CycleOptions:
    rounding:str; use_business_time:bool; workday_hours:int

def compute_cycle_time(start,end,dur_text,opts:CycleOptions):
    td=None
    if start and end: td=business_hours_diff(start,end,opts.workday_hours) if opts.use_business_time else (end-start)
    elif dur_text: td=parse_duration_text(dur_text)
    if td and opts.rounding in ("hours","days"): td=ceil_timedelta(td,opts.rounding)
    return td

def td_to_hours(td): return td.total_seconds()/3600 if td else None

def sidebar_options()->CycleOptions:
    st.sidebar.header("Options")
    rounding=st.sidebar.selectbox("Rounding",["days","hours","none"],0)
    use_business=st.sidebar.checkbox("Use business time (Mon–Fri only)",True)
    hours=st.sidebar.number_input("Workday hours",1,24,DEFAULT_WORKDAY_HOURS)
    return CycleOptions(rounding,use_business,hours)

def show_metrics(df,col):
    vals=df[col].dropna().values
    if len(vals)==0: return st.info("No values")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Median",f"{np.percentile(vals,50):.1f} h")
    c2.metric("P85",f"{np.percentile(vals,85):.1f} h")
    c3.metric("P95",f"{np.percentile(vals,95):.1f} h")
    c4.metric("Average",f"{np.mean(vals):.1f} h")

def charts(df,col,id_col,start_col=None):
    st.altair_chart(alt.Chart(df.dropna(subset=[col])).mark_bar().encode(
        x=alt.X(f"{col}:Q",bin=alt.Bin(maxbins=30),title="Cycle Time (h)"),
        y="count()",tooltip=[col]).properties(height=300),use_container_width=True)
    if start_col and start_col in df.columns:
        sdf=df.dropna(subset=[col,start_col]).copy()
        sdf[start_col]=pd.to_datetime(sdf[start_col],errors="coerce")
        st.altair_chart(alt.Chart(sdf).mark_circle(size=60).encode(
            x=f"{start_col}:T",y=f"{col}:Q",tooltip=[id_col,col,start_col]
        ).properties(height=300),use_container_width=True)

def timepiece_section():
    st.subheader("Timepiece — Async Export (auto-discovery)")
    default_tis="https://tis.obss.io"
    default_jira=st.session_state.get("guessed_jira","https://your-domain.atlassian.net")
    api_host=st.text_input("API Host", default_tis, help="Try your Jira base if export 404s, e.g. https://your-domain.atlassian.net")
    token=st.text_input("TISJWT Token", type="password")

    colh1,colh2=st.columns(2)
    if colh1.button("Set host to tis.obss.io"):
        st.session_state["api_host"]=default_tis; st.experimental_rerun()
    if colh2.button("Set host to Jira base"):
        st.session_state["api_host"]=default_jira; st.experimental_rerun()

    preset=st.radio("Source",["Saved filter (filterId)","Project key","Custom JQL"],0,horizontal=True)
    c1,c2,c3=st.columns(3)
    page_size=c1.number_input("Page size",1,1000,25)
    aggregation_type=c2.selectbox("aggregationType",["average","sum","median","standardDeviation"],0)
    columns=["Issue Key","Time in Status (hours)"]
    params={"page":1,"pageSize":int(page_size),"columnsBy":"statusduration","aggregationType":aggregation_type,"columns":columns}
    if preset=="Saved filter (filterId)":
        fid=c3.number_input("Filter ID",1,100000000,10547); params.update({"filterType":"jqlfilter","filterId":int(fid)})
    elif preset=="Project key":
        pkey=c3.text_input("Project Key","YOURPROJ"); params.update({"filterType":"project","projectKey":pkey})
    else:
        jql=c3.text_input("JQL","project = YOURPROJ AND updated >= -14d"); params.update({"filterType":"customjql","jql":jql})
    st.markdown("**Params preview**"); st.code(json.dumps(params,indent=2))

    if not token:
        st.info("Enter your TISJWT token to enable fetching."); return

    client=TimepieceClient(api_host if "api_host" not in st.session_state else st.session_state["api_host"], token)

    cA,cB=st.columns(2)
    if cA.button("Find export endpoint"):
        found = client.discover_export_endpoint(sample_params=params)
        if found:
            st.success(f"Export endpoint found: {found}")
        else:
            st.error("No export endpoint variant responded (non-404). Try switching API Host above.")
    if cB.button("Export (CSV) — recommended"):
        try:
            csv_bytes=client.export_flow_fetch_csv(params, st=st)
            df=pd.read_csv(io.BytesIO(csv_bytes))
            st.success("Export OK ✅"); st.dataframe(df.head(50))
            if "Time in Status (hours)" in df.columns:
                df["CycleTime_hours"]=df["Time in Status (hours)"].astype(float)
                show_metrics(df,"CycleTime_hours"); charts(df,"CycleTime_hours","Issue Key")
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "timepiece_export.csv", "text/csv")
        except Exception as ex:
            st.error(f"Export failed: {ex}")

def main():
    st.title("Cycle Time Tracker")
    tab = st.tabs(["Timepiece"])[0]
    with tab: timepiece_section()

if __name__=="__main__": main()
