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
    st.subheader("Timepiece")
    api_host=st.text_input("API Host","https://tis.obss.io")
    token=st.text_input("TISJWT Token",type="password")
    auth_mode=st.radio("Auth mode",["Header (Authorization: TISJWT ...)","Query (?token=...)"],0,horizontal=True)
    st.markdown("#### Source preset")
    preset=st.radio("Source",["Saved filter (filterId)","Project key","Custom JQL"],0,horizontal=True)
    c1,c2=st.columns(2)
    safe_mode=c1.checkbox("Safe Mode (minimal columns, small pages)",True)
    page_size=c2.number_input("Page size",1,1000,25 if safe_mode else 100)
    columns_by="statusduration"
    columns=["Issue Key","Time in Status (hours)"] if safe_mode else ["Issue Key","Issue Summary","Time in Status (hours)"]
    params={"page":1,"pageSize":int(page_size),"columnsBy":columns_by,"columns":columns}
    if preset=="Saved filter (filterId)":
        fid=st.number_input("Filter ID",1,100000000,12345); params.update({"filterType":"jqlfilter","filterId":int(fid)})
    elif preset=="Project key":
        pkey=st.text_input("Project Key","YOURPROJ"); params.update({"filterType":"project","projectKey":pkey})
    else:
        jql=st.text_input("JQL","project = YOURPROJ AND updated >= -14d"); params.update({"filterType":"customjql","jql":jql})
    st.markdown("**Params preview**"); st.code(json.dumps(params,indent=2))

    colA,colB,colC=st.columns(3)
    fetch_csv=colA.button("Fetch (CSV)")
    fetch_json=colB.button("Fetch (JSON)")
    test_btn=colC.button("Test Connection")

    if not token:
        st.info("Enter your TISJWT token to enable fetching."); return

    client=TimepieceClient(api_host,token,auth_mode=("query" if "Query" in auth_mode else "header"))

    if test_btn:
        try:
            test_params=dict(params); test_params["pageSize"]=1
            data=client.aggregation_json(test_params)
            st.success("Connection OK ✅")
            st.caption(f"Returned keys: {list(data)[:6]}")
        except Exception as ex:
            st.error(f"Test failed: {ex}")

    def handle_df(df: pd.DataFrame):
        if df.empty: return st.warning("No rows returned.")
        st.dataframe(df.head(50))
        num_cols=[c for c in df.columns if df[c].dtype.kind in "iuf"]
        default=[c for c in num_cols if "Time in Status" in c][:1]
        dcols=st.multiselect("Select duration columns to sum",options=num_cols,default=default)
        units=st.selectbox("Units of selected columns",["hours","days"],0)
        if dcols:
            df["_CycleTimeSum"]=df[dcols].sum(axis=1).astype(float)
            df["CycleTime_hours"]=df["_CycleTimeSum"]*(24.0 if units=="days" else 1.0)
            if opts.rounding in ("hours","days"):
                mult=1.0 if opts.rounding=="hours" else 24.0
                df["CycleTime_hours"]=np.ceil(df["CycleTime_hours"]/mult)*mult
            show_metrics(df,"CycleTime_hours")
            id_col="Issue Key" if "Issue Key" in df.columns else df.columns[0]
            date_like=[c for c in df.columns if any(k in c.lower() for k in ["start","created","begin"])]
            charts(df,"CycleTime_hours",id_col,start_col=(date_like[0] if date_like else None))
            st.download_button("Download CSV",df.to_csv(index=False).encode("utf-8"),"timepiece_cycle_time.csv","text/csv")

    def fetch_as_json(p):
        return client.aggregation_json_df(p)

    if fetch_csv:
        try:
            df=pd.read_csv(io.BytesIO(client.aggregation_csv(params)))
            handle_df(df)
        except Exception as ex:
            st.error(f"CSV failed: {ex}")
            try:
                st.info("Retrying as JSON for more details...")
                df=fetch_as_json(params)
                handle_df(df)
            except Exception as ex2:
                st.error(f"JSON fallback also failed: {ex2}")

    if fetch_json:
        try:
            df=fetch_as_json(params)
            handle_df(df)
        except Exception as ex:
            st.error(f"JSON failed: {ex}")

def main():
    st.title("Cycle Time Tracker")
    opts=sidebar_options()
    tab1,tab2=st.tabs(["Upload","Timepiece"])
    with tab1: upload_section(opts)
    with tab2: timepiece_section(opts)

if __name__=="__main__": main()
