import requests
class TimepieceClient:
    def __init__(self, api_host:str, token:str, timeout:int=60):
        self.host=api_host.rstrip("/"); self.session=requests.Session()
        self.session.headers.update({"Authorization":f"TISJWT {token.strip()}"})
        self.timeout=timeout
    def aggregation_csv(self,params:dict)->bytes:
        r=self.session.post(f"{self.host}/rest/aggregation",json=params,timeout=self.timeout,headers={"Accept":"text/csv"})
        if r.status_code>=400: raise RuntimeError(f"CSV failed {r.status_code}: {r.text[:200]}")
        return r.content
    def aggregation_json(self,params:dict)->dict:
        r=self.session.post(f"{self.host}/rest/aggregation",json=params,timeout=self.timeout,headers={"Accept":"application/json"})
        if r.status_code>=400: raise RuntimeError(f"JSON failed {r.status_code}: {r.text[:200]}")
        return r.json()