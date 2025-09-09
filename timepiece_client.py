import requests, pandas as pd

class TimepieceClient:
    def __init__(self, api_host:str, token:str, auth_mode:str="header", timeout:int=60):
        self.host=api_host.rstrip("/"); self.timeout=timeout; self.auth_mode=auth_mode
        self.session=requests.Session()
        if auth_mode=="header":
            self.session.headers.update({"Authorization":f"TISJWT {token.strip()}"})
        self.token=token.strip()

    def _url(self, path:str)->str:
        if self.auth_mode=="query":
            sep = "&" if "?" in path else "?"
            return f"{self.host}{path}{sep}token={self.token}"
        return f"{self.host}{path}"

    def aggregation_csv(self,params:dict)->bytes:
        r=self.session.post(self._url("/rest/aggregation"),json=params,timeout=self.timeout,
                            headers={"Accept":"text/csv","Content-Type":"application/json"})
        if r.status_code>=400: raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.content

    def aggregation_json(self,params:dict)->dict:
        r=self.session.post(self._url("/rest/aggregation"),json=params,timeout=self.timeout,
                            headers={"Accept":"application/json","Content-Type":"application/json"})
        if r.status_code>=400: raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.json()

    def aggregation_json_df(self, params:dict)->pd.DataFrame:
        data=self.aggregation_json(params)
        if isinstance(data, dict) and "rows" in data and "columns" in data:
            cols=[c.get("name") if isinstance(c,dict) else str(c) for c in data["columns"]]
            return pd.DataFrame(data["rows"], columns=cols)
        return pd.json_normalize(data)
