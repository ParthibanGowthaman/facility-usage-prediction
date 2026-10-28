# Facility Usage Prediction System

[![CI](https://github.com/ParthibanGowthaman/facility-usage-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/ParthibanGowthaman/facility-usage-prediction/actions)

For a residential community's shared facilities, predict a resident's next booking: **facility, usage day,
usage hour and notification (nudge) time**. Synthetic dataset, leakage-safe features, a chronological holdout, and a
spreadsheet comparing every prediction with the actual booking.

Full write-up: [`docs/TECHNICAL_DOC.pdf`](docs/TECHNICAL_DOC.pdf) (also `.docx` and `.md`, same content). Review spreadsheet: [`outputs/prediction_review.xlsx`](outputs/prediction_review.xlsx).

## Run it

Python 3.9+.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

python -m facility_pred.data_gen      # (optional) regenerate data/bookings.csv, seed 42
python -m facility_pred.data_report   # (optional) sanity checks on the dataset
python -m facility_pred.run all       # train on the early period, score the later holdout
pytest -q                             # leakage and reproducibility tests
```

`run all` runs three steps, which can also be run separately: `run train` (fits and saves `models/`), `run predict`
(writes `outputs/predictions.csv` for the holdout period, without looking at actuals) and `run evaluate` (compares with
the actuals; writes `results/` and `outputs/prediction_review.xlsx` + `.csv`).
Results are deterministic: rerunning reproduces `results/metrics.json` exactly.

## Result on the held-out period (2,736 bookings, from 2025-11-01)

| | Facility | Day | Hour | Nudge | Mean k of 4 | All 4 |
|---|---|---|---|---|---|---|
| Model | 45.1% | 42.5% | 39.6% | 17.4% | 36.1% | 3.7% |
| Baseline: resident's own habit | 47.6% | 27.5% | 39.8% | 12.7% | 31.9% | 2.9% |

The model wins overall and clearly on usage day and nudge, but a simple per-resident habit rule is slightly better on facility and
ties on hour. Nudge hit = nudge weekday correct and booking time within 30 minutes. See the technical document for limitations.

## Layout

```
data/        bookings.csv, facilities.csv (synthetic, seeded)
src/facility_pred/
  data_gen.py, data_report.py   dataset generator and its sanity report
  features.py                   leakage-safe features and the chronological split
  models.py                     gradient-boosted classifiers, clock-time rule, baselines
  evaluate.py, review.py        metrics / error analysis, review workbook
  run.py                        train / evaluate / all
models/ results/ outputs/       generated artifacts (committed)
tests/                          leakage, split and determinism tests
```
