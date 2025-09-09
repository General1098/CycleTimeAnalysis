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

def upload_section(opts):
    st.subheader("Upload CSV/Excel")
    file=st.file_uploader("Upload",type=["csv","xlsx"])
    if not file: return
    df=pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    id_col="ID" if "ID" in df.columns else df.columns[0]
    def row_td(r): return compute_cycle_time(to_datetime_safe(r.get("Start")),to_datetime_safe(r.get("End")),r.get("DurationText"),opts)
    df["CycleTime_hours"]=df.apply(row_td,axis=1).apply(td_to_hours)
    st.dataframe(df.head(50)); show_metrics(df,"CycleTime_hours"); charts(df,"CycleTime_hours",id_col,"Start")

def timepiece_section(opts):
    st.subheader("Timepiece — Async Export (default)")
    api_host=st.text_input("API Host","https://tis.obss.io")
    token=st.text_input("TISJWT Token",type="password")
    st.markdown("#### Source & aggregation")
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

    # Default client: query-string params (known-good), export fallback enabled
    client=TimepieceClient(api_host, token, auth_mode="header", send_params="Query string", max_retries=3, base_delay=1.0, export_fallback=True)

    # Primary path: Async Export
    if st.button("Export (CSV) — recommended"):
        try:
            csv_bytes=client.export_flow_fetch_csv(params, st=st)
            df=pd.read_csv(io.BytesIO(csv_bytes))
            st.success("Export OK ✅"); st.dataframe(df.head(50))
            # Map duration column if present
            if "Time in Status (hours)" in df.columns:
                df["CycleTime_hours"]=df["Time in Status (hours)"].astype(float)
                show_metrics(df,"CycleTime_hours"); charts(df,"CycleTime_hours","Issue Key")
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "timepiece_export.csv", "text/csv")
        except Exception as ex:
            st.error(f"Export failed: {ex}")

    with st.expander("Advanced — Live aggregation (may hit 429)"):
        cA,cB,cC = st.columns(3)
        auth_mode=cA.selectbox("Auth",["Header","Query"],0)
        send_params=cB.selectbox("Send params",["Query string","JSON body","Form-encoded"],0)
        retries=int(cC.number_input("Max retries on 429",1,12,6))
        if st.button("Fetch (JSON)"):
            try:
                live=TimepieceClient(api_host, token, auth_mode=("query" if auth_mode=="Query" else "header"), send_params=send_params, max_retries=retries, base_delay=1.5, export_fallback=False)
                df=live.fetch_json_with_robust_429(params, st=st)
                st.success("JSON OK ✅"); st.dataframe(df.head(50))
            except Exception as ex:
                st.error(f"JSON failed: {ex}")
        if st.button("Fetch (CSV)"):
            try:
                live=TimepieceClient(api_host, token, auth_mode=("query" if auth_mode=="Query" else "header"), send_params=send_params, max_retries=retries, base_delay=1.5, export_fallback=False)
                csv_bytes=live.fetch_csv_with_robust_429(params, st=st)
                df=pd.read_csv(io.BytesIO(csv_bytes))
                st.success("CSV OK ✅"); st.dataframe(df.head(50))
            except Exception as ex:
                st.error(f"CSV failed: {ex}")

def main():
    st.title("Cycle Time Tracker")
    opts=sidebar_options()
    tab1,tab2=st.tabs(["Upload","Timepiece"])
    with tab1: upload_section(opts)
    with tab2: timepiece_section(opts)

if __name__=="__main__": main()
