# Cycle Time Tracker (Streamlit) — with Timepiece integration

## Features
- Parse durations like `0d 2h 36m 50s` or compute from Start/End.
- Rounding: ceil to days (default), hours, or none.
- Optional business-time (Mon–Fri, configurable hours).
- Pull **Timepiece (Time in Status)** via TISJWT token.
- Charts: histogram + scatter, percentile metrics.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Timepiece
Create `.streamlit/secrets.toml` (or paste token in the UI):
```toml
[timepiece]
api_host = "https://tis.obss.io"
api_token = "YOUR_TISJWT_TOKEN"
```
Then open the **Timepiece** tab and provide aggregation params (JQL or filterId and columns). Start minimal:
```json
{
  "page": 1,
  "pageSize": 50,
  "columns": ["Issue Key", "Time in Status (hours)"],
  "jql": "project = ABC AND updated >= -14d"
}
```

## Repo tips
- Do **not** commit `.streamlit/secrets.toml` with real tokens.
- Add a `.gitignore` to exclude `.venv/` and `__pycache__/`.
