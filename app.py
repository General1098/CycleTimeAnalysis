import io, json, pandas as pd, streamlit as st
from timepiece_client import TimepieceClient

def timepiece_section():
    st.subheader("Timepiece — Export auto-discovery")
    api_host = st.text_input("API Host","https://tis.obss.io")
    token = st.text_input("TISJWT Token", type="password")
    if not token: return
    client = TimepieceClient(api_host, token)
    params = {"filterType":"jqlfilter","filterId":10547,"columnsBy":"statusduration","aggregationType":"average","page":1,"pageSize":25,"columns":["Issue Key","Time in Status (hours)"]}
    st.code(json.dumps(params, indent=2))
    if st.button("Export (CSV)"):
        try:
            csv_bytes=client.export_flow_fetch_csv(params, st=st)
            df=pd.read_csv(io.BytesIO(csv_bytes))
            st.dataframe(df.head())
        except Exception as ex:
            st.error(str(ex))

def main():
    st.title("Cycle Time Tracker v15")
    timepiece_section()

if __name__=="__main__": main()