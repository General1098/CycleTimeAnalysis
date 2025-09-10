from __future__ import annotations
import io, json
import pandas as pd
import streamlit as st

from timepiece_client import TimepieceClient

DEFAULT_PARAMS = {
  "filterType": "jqlfilter",
  "jqlFilterID": 10547,
  "columnsBy": "statusDuration",
  "aggregationType": "average",
  "page": 1,
  "pageSize": 25,
  "columns": ["Issue Key", "Time in Status (hours)"]
}

def timepiece_tab():
    st.header("Timepiece — /rest API (v17)")
    api_host = st.text_input("API Host", "https://tis.obss.io")
    token = st.text_input("TISJWT Token", type="password")
    params_json = st.text_area(
        "Report params (JSON)", 
        json.dumps(DEFAULT_PARAMS, indent=2), 
        height=180
    )
    try:
        params_obj = json.loads(params_json)
    except Exception as e:
        st.error(f"Params JSON invalid: {e}")
        return

    client = TimepieceClient(api_host, token)

    st.subheader("Sync Aggregation (/rest/aggregation)")
    if st.button("Fetch JSON"):
        try:
            df = client.aggregation_json_df(params_obj)
            st.success("OK")
            st.dataframe(df.head(50))
        except Exception as ex:
            st.error(f"JSON failed: {ex}")
    if st.button("Fetch CSV"):
        try:
            csv_bytes = client.aggregation_csv(params_obj)
            df = pd.read_csv(io.BytesIO(csv_bytes))
            st.success("OK")
            st.dataframe(df.head(50))
        except Exception as ex:
            st.error(f"CSV failed: {ex}")

    st.subheader("Async Export (/rest/export)")
    agg_choice = st.selectbox(
        "Aggregation for CSV export", 
        ["average", "sum", "median", "standardDeviation"], 
        0
    )
    if st.button("Start Export + Poll + Download"):
        try:
            df = client.export_to_dataframe(params_obj, agg_choice)
            st.success("Export OK")
            st.dataframe(df.head(50))
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "timepiece_export.csv",
                "text/csv"
            )
        except Exception as ex:
            st.error(f"Export failed: {ex}")

def main():
    timepiece_tab()

if __name__ == "__main__":
    main()
