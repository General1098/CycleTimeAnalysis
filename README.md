# Cycle Time Tracker with Timepiece (v6)

**New in v6**
- Safe Mode (minimal columns, small pages) to avoid 500s
- JSON fetch + automatic CSV→JSON fallback for better diagnostics
- Presets UI (Saved filter / Project / Custom JQL)

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Timepiece usage
- Keep `columnsBy` as `statusduration`.
- Start with:
```json
{
  "filterType": "jqlfilter",
  "filterId": 12345,
  "columnsBy": "statusduration",
  "page": 1,
  "pageSize": 25,
  "columns": ["Issue Key", "Time in Status (hours)"]
}
```
