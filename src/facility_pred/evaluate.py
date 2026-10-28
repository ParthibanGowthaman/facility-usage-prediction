"""Metrics for the four outputs, an overall score, and error analysis. Used on the test split only."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

NUDGE_TOL_MIN = 30


def hits(X, P):
    """Per-record hit indicators. P has p_fac, p_dow, p_hour, p_nudge_dow, p_clock."""
    h = pd.DataFrame(index=X.index)
    h["facility"] = (P.p_fac.values == X.y_fac.values)
    h["day"] = (P.p_dow.values == X.y_dow.values)
    h["hour"] = (P.p_hour.values == X.y_hour.values)
    clock_ok = np.abs(P.p_clock.values - X.y_clock.values) <= NUDGE_TOL_MIN
    h["nudge"] = (P.p_nudge_dow.values == X.y_nudge_dow.values) & clock_ok
    h["k_of_4"] = h[["facility", "day", "hour", "nudge"]].sum(axis=1)
    h["all_4"] = h.k_of_4 == 4
    return h


def metrics(X, P, top3=None):
    h = hits(X, P)
    circ = np.abs(P.p_dow.values - X.y_dow.values)
    circ = np.minimum(circ, 7 - circ)
    clock_err = np.abs(P.p_clock.values - X.y_clock.values)
    m = {
        "facility": {"accuracy": h.facility.mean(),
                     "macro_f1": f1_score(X.y_fac, P.p_fac, average="macro"),
                     **({"top3_accuracy": float(np.mean([y in t for y, t in zip(X.y_fac, top3)]))}
                        if top3 is not None else {})},
        "day": {"accuracy": h.day.mean(), "within_1_day": float((circ <= 1).mean())},
        "hour": {"accuracy": h.hour.mean(),
                 "within_1_hour": float((np.abs(P.p_hour.values - X.y_hour.values) <= 1).mean()),
                 "mae_hours": float(np.abs(P.p_hour.values - X.y_hour.values).mean())},
        "nudge": {"accuracy": h.nudge.mean(),
                  "day_accuracy": float((P.p_nudge_dow.values == X.y_nudge_dow.values).mean()),
                  f"clock_within_{NUDGE_TOL_MIN}min": float((clock_err <= NUDGE_TOL_MIN).mean()),
                  "clock_within_60min": float((clock_err <= 60).mean()),
                  "clock_mae_min": float(clock_err.mean())},
        "overall": {"mean_k_of_4": h.k_of_4.mean() / 4, "all_4_match": h.all_4.mean()},
    }
    return {k: {a: float(b) for a, b in v.items()} for k, v in m.items()}


def segment_table(X, h):
    """Accuracy by how much history the resident had, and by true facility."""
    d = pd.concat([X[["n_prev", "y_fac"]], h], axis=1)
    d["history"] = pd.cut(d.n_prev, [-1, 0, 9, 29, 10**6], labels=["0 (cold start)", "1-9", "10-29", "30+"])
    by_hist = d.groupby("history", observed=True).agg(
        n=("k_of_4", "size"), facility=("facility", "mean"), day=("day", "mean"),
        hour=("hour", "mean"), nudge=("nudge", "mean"), mean_k_of_4=("k_of_4", lambda s: s.mean() / 4))
    by_fac = d.groupby("y_fac").agg(n=("k_of_4", "size"), facility=("facility", "mean"), day=("day", "mean"),
                                    hour=("hour", "mean"), nudge=("nudge", "mean"))
    return by_hist, by_fac
