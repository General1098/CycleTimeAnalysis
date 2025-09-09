# Cycle Time Tracker (Streamlit) — Timepiece Ready

**Features**
- Parse durations like `0d 2h 36m 50s` or compute from Start/End.
- Rounding: ceil to days (default), hours, or none.
- Optional business-time (Mon–Fri only, configurable hours).
- Pull Timepiece (Time in Status) data via TISJWT.
- Charts: histogram + scatter, percentile metrics.

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Configure Secrets
Create `.streamlit/secrets.toml`:
```toml
[timepiece]
api_host = "https://tis.obss.io"
api_token = "YOUR_TISJWT_TOKEN"

[app]
workday_hours = 8
default_rounding = "days"
use_business_time = true
```

## Timepiece Aggregation
In the **Timepiece** tab, provide parameters like:
```json
{
  "page": 1,
  "pageSize": 100,
  "columns": ["Issue Key", "Issue Summary", "Assignee", "Time in Status (hours)"],
  "jql": "project = ABC AND statusCategory != Done"
}
```
Select which numeric columns to sum as **Cycle Time** (e.g., specific status hour columns).
