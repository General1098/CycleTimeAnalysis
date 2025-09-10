import io, json, pandas as pd, streamlit as st
from timepiece_client import TimepieceClient

DEFAULT_PARAMS = {
  "filterType":"jqlfilter",
  "filterId":10547,
  "columnsBy":"statusduration",
  "aggregationType":"average",
  "page":1,
  "pageSize":25,
  "columns":["Issue Key","Time in Status (hours)"]
}

def timepiece_section():
    st.subheader("Timepiece — Export (manual override + discovery)")
    api_host = st.text_input("API Host", st.session_state.get("api_host","https://tis.obss.io"))
    st.session_state["api_host"] = api_host
    token = st.text_input("TISJWT Token", type="password")
    params = st.text_area("Report params (JSON)", json.dumps(DEFAULT_PARAMS, indent=2), height=180)
    try:
        params_obj = json.loads(params)
    except Exception as e:
        st.error(f"Params JSON invalid: {e}")
        return

    client = TimepieceClient(api_host, token)

    st.markdown("#### Export endpoint settings")
    use_custom = st.checkbox("Use custom export paths (override discovery)", False)
    c1,c2 = st.columns(2)
    if use_custom:
        start_path = c1.text_input("Start path", st.session_state.get("start_path","/report/export"))
        status_path = c2.text_input("Status path (use {id})", st.session_state.get("status_path","/report/export/{id}"))
        download_path = st.text_input("Download path (use {id})", st.session_state.get("download_path","/report/export/{id}/download"))
        st.session_state.update({"start_path":start_path,"status_path":status_path,"download_path":download_path})
        client.set_custom_paths(start_path, status_path, download_path)

    colA,colB,colC = st.columns(3)
    if colA.button("Discover export endpoint"):
        found = client.discover_export_endpoint(sample_params=params_obj)
        if found:
            st.success(f"Discovered start: {found}")
        else:
            st.error("No export endpoint responded (non-404). Try switching API Host to your Jira base URL.")
    if colB.button("Export (CSV)"):
        try:
            csv_bytes = client.export_flow_fetch_csv(params_obj, st=st)
            df = pd.read_csv(io.BytesIO(csv_bytes))
            st.success("Export OK ✅"); st.dataframe(df.head(50))
            if "Time in Status (hours)" in df.columns:
                df["CycleTime_hours"] = df["Time in Status (hours)"].astype(float)
        except Exception as ex:
            st.error(f"Export failed: {ex}")
    if colC.button("Show cURL for Start"):
        st.code(client.curl_preview_start(params_obj))

    st.markdown("#### Probe endpoints")
    colP1,colP2,colP3 = st.columns(3)
    if colP1.button("Ping Start (POST)"):
        code, text = client.ping_start(params_obj)
        st.write(f"HTTP {code}")
        st.code(text[:600])
    eid = st.text_input("Export ID (for status/download pings)", "")
    if colP2.button("Ping Status (GET)"):
        code, text = client.ping_status(eid or "TEST-ID")
        st.write(f"HTTP {code}")
        st.code(text[:600])
    if colP3.button("Ping Download (GET)"):
        code, text = client.ping_download(eid or "TEST-ID")
        st.write(f"HTTP {code}")
        st.code(text[:600])

def main():
    st.title("Cycle Time Tracker v16")
    timepiece_section()

if __name__=="__main__": main()