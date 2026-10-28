# Facility Usage Prediction System

[![CI](https://github.com/ParthibanGowthaman/facility-usage-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/ParthibanGowthaman/facility-usage-prediction/actions)

For a residential community's shared facilities, predict a resident's next booking: **facility, usage day,
usage hour and notification (nudge) time**. Synthetic dataset, leakage-safe features, a chronological holdout, and a
spreadsheet comparing every prediction with the actual booking.

Full write-up: [`docs/TECHNICAL_DOC.md`](docs/TECHNICAL_DOC.md). Review spreadsheet: [`outputs/prediction_review.xlsx`](outputs/prediction_review.xlsx).

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

`run all` writes `models/` (fitted models and chosen hyper-parameters), `results/` (metrics and tables) and
`outputs/prediction_review.xlsx` (+ `.csv`). `run train` and `run evaluate` run the two halves separately.
Results are deterministic: rerunning reproduces `results/metrics.json` exactly.

## Result on the held-out period (2,572 bookings, from 2025-11-01)

| | Facility | Day | Hour | Nudge | Mean k of 4 | All 4 |
|---|---|---|---|---|---|---|
| Model | 45.6% | 40.1% | 38.3% | 15.4% | 34.9% | 4.1% |
| Baseline: resident's own habit | 48.8% | 27.1% | 40.9% | 11.5% | 32.1% | 3.3% |

The model wins overall and clearly on usage day, but a simple per-resident habit rule is slightly better on facility and
hour. Nudge hit = nudge weekday correct and booking time within 30 minutes. See the technical document for limitations.

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
