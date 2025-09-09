# Cycle Time Analysis — Timepiece Integration

This repo version matches your current app and adds a **Timepiece (Time in Status)** integration so you can fetch status-duration data without parsing Jira changelogs.

## What's included
- `app.py` — your Cycle Time Analysis app with a new **Timepiece** tab:
  - Enter **API Host** (usually `https://tis.obss.io`) and your **TISJWT** token.
  - Paste **Aggregation Parameters (JSON)** (JQL/filter & columns).
  - Click **Fetch** to load data directly into the app.
  - Safe Mode (smaller pages, minimal columns) and CSV→JSON fallback for robustness.
- `timepiece_client.py` — minimal client using `Authorization: TISJWT <token>` supporting:
  - `/rest/aggregation` (CSV & JSON)
  - `/report/export` async flow (start, poll, download)
- `.streamlit/secrets-template.toml` — template for local secrets
- `.streamlit/config.toml` — optional theme settings

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Configure Secrets (optional)
Create `.streamlit/secrets.toml`:
```toml
[timepiece]
api_host = "https://tis.obss.io"
api_token = "YOUR_TISJWT_TOKEN"
```

## Minimal Aggregation Params example
```json
{
  "page": 1,
  "pageSize": 50,
  "columns": ["Issue Key", "Time in Status (hours)"],
  "jql": "project = ABC AND updated >= -14d"
}
```
Then map the numeric columns to **Cycle Time** in the UI.
