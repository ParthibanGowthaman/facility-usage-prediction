"""Leakage-safe feature building and the chronological split.

Framing: for every booking, predict the resident's NEXT booking from everything the resident did
up to and including their previous booking. Prediction time T is the previous booking's
`booking_ts`. Features are built by a single forward pass over each resident's bookings in
`booking_ts` order: a row's features are emitted first, and only then is the row folded into the
resident's running state. A row's features therefore cannot depend on that row or any later one.
"""
import numpy as np
import pandas as pd

N_FAC, DECAY = 6, 0.9
TRAIN_END, VAL_END = pd.Timestamp("2025-09-15"), pd.Timestamp("2025-11-01")  # test = booking_ts >= VAL_END
LEAD_CAP = 7  # calendar days between booking and usage, clipped

TARGETS = ["y_fac", "y_dow", "y_hour", "y_lead", "y_clock", "y_nudge_dow"]


def load(data_dir="data"):
    b = pd.read_csv(f"{data_dir}/bookings.csv", parse_dates=["booking_ts", "usage_ts"])
    b["fac"] = b.facility_id.str[1:].astype(int) - 1
    return b.sort_values(["booking_ts", "booking_id"]).reset_index(drop=True)


def split_of(booking_ts):
    return np.where(booking_ts < TRAIN_END, "train", np.where(booking_ts < VAL_END, "val", "test"))


def _mode(hist):
    return float(np.argmax(hist)) if hist.sum() > 0 else np.nan


class _State:
    """Running per-resident history. Decayed ('ewma') histograms favour recent behaviour."""

    def __init__(self):
        self.n = 0
        self.fac_all = np.zeros(N_FAC)
        self.fac_ew = np.zeros(N_FAC)
        self.dow_ew = np.zeros(7)
        self.bdow_ew = np.zeros(7)  # weekday on which the resident tends to BOOK
        self.lead_ew = np.zeros(LEAD_CAP + 1)
        self.hour_ew = np.zeros(24)
        self.hour_by_fac = np.zeros((N_FAC, 24))
        self.dow_by_fac = np.zeros((N_FAC, 7))
        self.clocks, self.book_ts, self.last = [], [], None

    def features(self):
        f = {"n_prev": self.n}
        if self.n == 0:
            return f  # cold start: everything else is NaN and the model falls back on global patterns
        l = self.last
        f.update(last_fac=l["fac"], last_dow=l["dow"], last_hour=l["hour"], last_lead=l["lead"],
                 last_clock=l["clock"], last_book_dow=l["book_dow"], last_lead_h=l["lead_h"],
                 first_seen_days=(self.book_ts[-1] - self.book_ts[0]) / 86400,
                 book_month=l["book_month"])
        gaps = np.diff(self.book_ts) / 3600
        f.update(gap_median_h=np.median(gaps) if len(gaps) else np.nan,
                 gap_last_h=gaps[-1] if len(gaps) else np.nan,
                 prev2_fac=l["prev2_fac"], prev3_fac=l["prev3_fac"])
        for i in range(N_FAC):
            f[f"fac_all_{i}"] = self.fac_all[i] / self.n
            f[f"fac_ew_{i}"] = self.fac_ew[i] / self.fac_ew.sum()
        for i in range(7):
            f[f"dow_ew_{i}"] = self.dow_ew[i] / self.dow_ew.sum()
        for i in range(LEAD_CAP + 1):
            f[f"lead_ew_{i}"] = self.lead_ew[i] / self.lead_ew.sum()
        for i in range(7):
            f[f"bdow_ew_{i}"] = self.bdow_ew[i] / self.bdow_ew.sum()
        for i in range(N_FAC):  # habits per facility; NaN when never used
            f[f"hour_mode_f{i}"] = _mode(self.hour_by_fac[i])
            f[f"dow_mode_f{i}"] = _mode(self.dow_by_fac[i])
        for h in range(24):
            f[f"hour_ew_{h}"] = self.hour_ew[h] / self.hour_ew.sum()
        order = np.argsort(-self.fac_ew)
        for rank, fi in enumerate(order[:2]):
            f[f"top{rank + 1}_fac"] = fi
            f[f"top{rank + 1}_fac_hour"] = _mode(self.hour_by_fac[fi])
            f[f"top{rank + 1}_fac_dow"] = _mode(self.dow_by_fac[fi])
        f["hour_mode"] = _mode(self.hour_ew)
        f["dow_mode"] = _mode(self.dow_ew)
        f["lead_mode"] = _mode(self.lead_ew)
        c = np.array(self.clocks[-10:])
        ck = np.array(self.clocks, dtype=float)
        w = DECAY ** np.arange(self.n)[::-1]
        dens = (w[None, :] * (np.abs(ck[:, None] - ck[None, :]) <= 30)).sum(1)  # recency-weighted mass within +/-30 min
        f.update(clock_mode=ck[np.argmax(dens)], clock_mode_share=dens.max() / w.sum())
        f.update(clock_med10=np.median(c), clock_std10=c.std(),
                 clock_ew=np.average(self.clocks, weights=DECAY ** np.arange(self.n)[::-1]))
        return f

    def update(self, r):
        d = DECAY
        self.fac_all[r.fac] += 1
        for arr in (self.fac_ew, self.dow_ew, self.bdow_ew, self.lead_ew, self.hour_ew):
            arr *= d
        self.fac_ew[r.fac] += 1
        self.dow_ew[r.dow] += 1
        self.bdow_ew[r.booking_ts.dayofweek] += 1
        self.lead_ew[r.lead] += 1
        self.hour_ew[r.hour] += 1
        self.hour_by_fac[r.fac] *= d
        self.hour_by_fac[r.fac, r.hour] += 1
        self.dow_by_fac[r.fac] *= d
        self.dow_by_fac[r.fac, r.dow] += 1
        ts = r.booking_ts.timestamp()
        clock = r.booking_ts.hour * 60 + r.booking_ts.minute
        self.clocks.append(clock)
        self.book_ts.append(ts)
        prev = self.last
        self.last = dict(fac=r.fac, dow=r.dow, hour=r.hour, lead=r.lead, clock=clock,
                         book_dow=r.booking_ts.dayofweek, book_month=r.booking_ts.month,
                         lead_h=(r.usage_ts - r.booking_ts).total_seconds() / 3600,
                         prev2_fac=prev["fac"] if prev else np.nan,
                         prev3_fac=prev["prev2_fac"] if prev else np.nan)
        self.n += 1


def _enrich(b):
    b = b.copy()
    b["dow"] = b.usage_ts.dt.dayofweek
    b["hour"] = b.usage_ts.dt.hour
    b["lead"] = (b.usage_ts.dt.normalize() - b.booking_ts.dt.normalize()).dt.days.clip(0, LEAD_CAP)
    b["clock"] = b.booking_ts.dt.hour * 60 + b.booking_ts.dt.minute
    return b


def build(bookings):
    """Return one row per booking: leakage-safe features, the 5 raw targets, and the split."""
    b = _enrich(bookings)
    states, rows = {}, []
    for r in b.itertuples(index=False):
        st = states.setdefault(r.resident_id, _State())
        rows.append(st.features())  # features first ...
        st.update(r)                # ... then fold this booking into history
    X = pd.DataFrame(rows)
    X.insert(0, "booking_id", b.booking_id.values)
    X.insert(1, "resident_id", b.resident_id.values)
    X["booking_ts"] = b.booking_ts.values
    X["split"] = split_of(b.booking_ts)
    X["y_fac"], X["y_dow"], X["y_hour"] = b.fac.values, b.dow.values, b.hour.values
    X["y_lead"], X["y_clock"] = b.lead.values, b.clock.values
    X["y_nudge_dow"] = b.booking_ts.dt.dayofweek.values
    return X


def feature_columns(X):
    meta = {"booking_id", "resident_id", "booking_ts", "split", *TARGETS}
    return [c for c in X.columns if c not in meta]
