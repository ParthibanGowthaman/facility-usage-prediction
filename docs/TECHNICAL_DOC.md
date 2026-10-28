# Facility Usage Prediction System: Technical Documentation

Sections are added as each slice lands. Status: **1. Data** complete.

## 1. Data

### 1.1 Why synthetic, and how it is built
The brief asks for realistic booking history for one residential community. No real data is available, so
`src/facility_pred/data_gen.py` simulates one calendar year (2025-01-01 to 2025-12-31) for 300 residents and
6 facilities. It is seeded (`--seed 42`), so the same command always produces the same file, and CI checks
that the regenerated `data/bookings.csv` is byte-identical to the committed one.

Schema (`data/bookings.csv`, one row per booking): `booking_id`, `resident_id`, `facility_id`,
`booking_ts` (when the booking was created), `usage_ts` (when the facility is used).
`data/facilities.csv` maps `facility_id` to name and operating hours.

### 1.2 Properties the brief asked for, and how each is produced

| Property | How it is simulated |
|---|---|
| Resident preferences | Each resident draws a sparse facility preference (Gamma-weighted popularity), 2-3 favourite weekdays, and a peak hour per facility. |
| Facility popularity / imbalance | Base shares: Gym 34%, Pool 22%, Badminton 16%, Tennis 12%, Clubhouse 9%, Hall 7%. |
| Time patterns | Peak hours differ by facility (gym 6-7am and 6-7pm; pool 7am and 5-6pm; etc.). Pool demand is x1.8 from March to September. Clubhouse and Hall demand is x2 at weekends. |
| Booking lead times | A resident-specific routine books at a habitual time of day, 0-3 days ahead (50-90% of their bookings). The rest are ad hoc with a log-normal lead time (median 14h, capped 0.5-240h). |
| Sparsity | Booking rate per resident is log-normal (heavy-tailed). About 30% of residents move in during the year and 12% leave. |
| Noise | 10% of bookings go to a random facility. Hours spread +/-1h around the peak, with 5% fully random. Booking time has a ~8 minute jitter. |
| Changing behaviour | 28% of residents switch to new facility preferences and shifted hours at a random changepoint. |

### 1.3 How it was verified
`python -m facility_pred.data_report` prints the distributions below, and a second ad hoc pass checked
the data directly. Results on the committed file (14,816 bookings, 300 residents):

- Integrity: no nulls, no duplicate booking IDs, no resident booked twice at the same time, every
  `booking_ts` is before its `usage_ts`, every usage hour is within the facility's opening hours.
- Imbalance: Gym 29%, Pool 23%, Badminton 14%, Clubhouse 12%, Tennis 11%, Hall 11% of bookings.
- Sparsity: median 40 bookings per resident (min 1, max 166); 7% of residents have fewer than 10.
- Preferences: a resident's favourite facility averages 52% of their bookings.
- Drift: 38% of residents with 30+ bookings have a different favourite facility in the second half of their history than in the first.
- Predictability: repeating the previous facility is right 38% of the time and the previous hour 28%, so simple baselines do not solve the task. Booking clock time is within +/-30 minutes of a resident's median in 61% of bookings, which caps how well the notification time can be predicted.
- Fix made during verification: a first version had all residents joining at random dates, so January had about 140 bookings against about 1,300 by August. Most residents (70%) are now present from the start, giving about 1,050 bookings in January and a flat monthly volume.

### 1.4 Limitations
- Results on this data show the pipeline works, not how it performs on real resident behaviour; the
  habits the models learn are the habits the generator planted.
- No facility capacity limits, cancellations, holidays, or multi-year seasonality.
- Resident habits are independent of each other; there are no household or group bookings.
