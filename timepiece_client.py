import json, time, requests
from typing import Dict, Any, Optional

class TimepieceClient:
    def __init__(self, api_host:str, token:str, timeout:int=60):
        self.host=api_host.rstrip("/"); self.timeout=timeout
        self.token=token.strip()
        self.session=requests.Session()
        self.session.headers.update({"Authorization":f"TISJWT {self.token}","User-Agent":"CycleTimeTracker/1.5"})
        self.export_candidates=[
            {"start":"/report/export","status":"/report/export/{id}","download":"/report/export/{id}/download"},
            {"start":"/rest/report/export","status":"/rest/report/export/{id}","download":"/rest/report/export/{id}/download"},
            {"start":"/api/report/export","status":"/api/report/export/{id}","download":"/api/report/export/{id}/download"},
            {"start":"/rest/obss-tis/latest/report/export","status":"/rest/obss-tis/latest/report/export/{id}","download":"/rest/obss-tis/latest/report/export/{id}/download"},
        ]
        self._chosen: Optional[dict]=None

    def _url(self,path:str)->str: return f"{self.host}{path}"

    def discover_export_endpoint(self,sample_params:Dict[str,Any])->Optional[str]:
        body={"params":sample_params,"format":"csv"}
        for cand in self.export_candidates:
            try:
                r=self.session.post(self._url(cand["start"]),json=body,timeout=self.timeout)
                if r.status_code!=404:
                    self._chosen=cand; return cand["start"]
            except Exception: continue
        return None

    def export_start(self,body:Dict[str,Any])->str:
        if not self._chosen: self.discover_export_endpoint(sample_params=body.get("params",{}))
        if not self._chosen: raise RuntimeError("No export endpoint found")
        r=self.session.post(self._url(self._chosen["start"]),json=body,timeout=self.timeout)
        if not r.ok: raise RuntimeError(f"{r.status_code}: {r.text}")
        data=r.json(); return str(data.get("id") or data.get("exportId") or "")

    def export_status(self,eid:str)->Dict[str,Any]:
        r=self.session.get(self._url(self._chosen["status"].format(id=eid)),timeout=self.timeout)
        if not r.ok: raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.json()

    def export_download(self,eid:str)->bytes:
        r=self.session.get(self._url(self._chosen["download"].format(id=eid)),timeout=self.timeout)
        if not r.ok: raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.content

    def export_flow_fetch_csv(self,params:Dict[str,Any],st=None)->bytes:
        body={"params":params,"format":"csv"}
        if st: st.info("Discovering export endpoint…")
        self.discover_export_endpoint(sample_params=params)
        if st: st.info("Starting export…")
        eid=self.export_start(body)
        if st: st.info(f"Export started: {eid}")
        for _ in range(30):
            time.sleep(2); info=self.export_status(eid)
            status=str(info.get("status") or info.get("state") or "").lower()
            if status in ("completed","done","finished","ready"):
                return self.export_download(eid)
        raise RuntimeError("Export timed out")