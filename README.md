# Facility Usage Prediction System

Predicts a resident's next facility booking: facility, usage day, usage hour and notification time.

Work in progress. Slice 1 (synthetic dataset) is complete.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
python -m facility_pred.data_gen      # writes data/bookings.csv, data/facilities.csv
python -m facility_pred.data_report   # sanity checks on the dataset
```
