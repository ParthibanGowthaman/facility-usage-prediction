"""Synthetic facility-booking data for one residential community.

Each resident gets latent habits (favourite facilities, weekdays, hours, booking
routine). A subset drifts to new habits at a changepoint, activity is heavy-tailed
(many sparse residents), facility popularity is imbalanced, and noise is mixed in.
Fully seeded: the same seed always produces the same dataset.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FACILITIES = [
    # id, name, popularity, open hour, close hour (last start hour), typical peak hours
    ("F1", "Gym", 0.34, 5, 22, [6, 7, 18, 19]),
    ("F2", "Pool", 0.22, 6, 21, [7, 17, 18]),
    ("F3", "Badminton Court", 0.16, 6, 22, [19, 20, 21]),
    ("F4", "Tennis Court", 0.12, 6, 21, [7, 8, 17, 18]),
    ("F5", "Clubhouse", 0.09, 10, 22, [11, 16, 19]),
    ("F6", "Multipurpose Hall", 0.07, 9, 22, [10, 15, 18]),
]
START, END = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")


def _hour_probs(rng, f, peak):
    """Hour distribution for one resident-facility: sharp peak, small spread, noise."""
    _, _, _, lo, hi, _ = f
    hours = np.arange(lo, hi + 1)
    p = np.full(len(hours), 0.05 / len(hours))
    for off, w in ((0, 0.72), (-1, 0.09), (1, 0.09)):
        if peak + off in hours:
            p[hours == peak + off] += w
    return hours, p / p.sum()


def _new_habits(rng, pop):
    prefs = pop * rng.gamma(0.45, size=len(pop))  # sparse: 1-2 favourites dominate
    prefs = prefs / prefs.sum()
    peaks = [int(rng.choice(f[5])) for f in FACILITIES]
    return prefs, peaks


def generate(seed=42, n_residents=300):
    rng = np.random.default_rng(seed)
    pop = np.array([f[2] for f in FACILITIES])
    days = pd.date_range(START, END, freq="D")
    rows = []

    for r in range(n_residents):
        rid = f"R-{101 + r}"
        rate = float(np.clip(rng.lognormal(np.log(1.1), 0.95), 0.05, 6.0))  # bookings/week
        fav_days = rng.choice(7, size=rng.integers(2, 4), replace=False)
        wd_w = np.full(7, 0.12)
        wd_w[fav_days] = 1.0
        wd_w /= wd_w.mean()

        join = START  # most residents are established; ~30% move in during the year
        if rng.random() < 0.3:
            join = START + pd.Timedelta(days=int(rng.uniform(0, 0.7) * len(days)))
        leave = END
        if rng.random() < 0.12:  # churn
            leave = join + pd.Timedelta(days=int(rng.uniform(60, 220)))

        prefs, peaks = _new_habits(rng, pop)
        drift_at = None
        if rng.random() < 0.28:  # behaviour change
            drift_at = join + pd.Timedelta(days=int(rng.uniform(0.35, 0.8) * (min(leave, END) - join).days))
            prefs2, peaks2 = _new_habits(rng, pop)
            peaks2 = [p if rng.random() < 0.3 else q for p, q in zip(peaks, peaks2)]
        habit_hour = int(rng.integers(8, 22))
        habit_min = int(rng.choice([0, 15, 30, 45]))
        lead_days_p = rng.dirichlet([2, 4, 2, 1])  # lead of 0..3 days
        habit_share = rng.uniform(0.5, 0.9)

        for d in days:
            if d < join or d > leave:
                continue
            wd = d.weekday()
            if rng.random() > 1 - np.exp(-rate / 7 * wd_w[wd]):
                continue
            after = drift_at is not None and d >= drift_at
            p_, pk = (prefs2, peaks2) if after else (prefs, peaks)
            w = p_.copy()
            if wd >= 5:
                w[4:] *= 2.0  # clubhouse/hall busier at weekends
            if 3 <= d.month <= 9:
                w[1] *= 1.8  # pool season
            w = 0.9 * w / w.sum() + 0.1 / len(w)  # noise: occasional off-habit facility
            fi = int(rng.choice(len(FACILITIES), p=w / w.sum()))
            hrs, hp = _hour_probs(rng, FACILITIES[fi], pk[fi])
            hour = int(rng.choice(hrs, p=hp))
            minute = int(rng.choice([0, 0, 0, 30]))
            usage = d + pd.Timedelta(hours=hour, minutes=minute)

            booking = None
            if rng.random() < habit_share:  # routine: book at the same time of day
                ld = int(rng.choice(4, p=lead_days_p))
                cand = (d - pd.Timedelta(days=ld)).normalize() + pd.Timedelta(
                    hours=habit_hour, minutes=habit_min + int(rng.normal(0, 8)))
                if cand <= usage - pd.Timedelta(minutes=30):
                    booking = cand
            if booking is None:  # ad-hoc: heavy-tailed lead time, made while awake (07:00-22:59)
                for _ in range(50):
                    lead_h = float(np.clip(rng.lognormal(np.log(14), 1.0), 0.5, 240))
                    booking = usage - pd.Timedelta(hours=lead_h)
                    if 7 <= booking.hour < 23:
                        break
                else:  # no waking-hours draw found: book the previous evening
                    booking = (usage.normalize() - pd.Timedelta(days=1)) + pd.Timedelta(hours=20)
            booking = booking.floor("min")
            if booking < START:
                continue
            rows.append((rid, FACILITIES[fi][0], booking, usage))

    df = pd.DataFrame(rows, columns=["resident_id", "facility_id", "booking_ts", "usage_ts"])
    df = df.sort_values(["booking_ts", "resident_id"]).reset_index(drop=True)
    df.insert(0, "booking_id", [f"B-{i + 1:06d}" for i in range(len(df))])
    fac = pd.DataFrame([(f[0], f[1], f[3], f[4]) for f in FACILITIES],
                       columns=["facility_id", "facility_name", "open_hour", "last_start_hour"])
    return df, fac


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--residents", type=int, default=300)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, fac = generate(a.seed, a.residents)
    df.to_csv(out / "bookings.csv", index=False)
    fac.to_csv(out / "facilities.csv", index=False)
    print(f"wrote {len(df)} bookings, {df.resident_id.nunique()} residents -> {out}/")


if __name__ == "__main__":
    main()
