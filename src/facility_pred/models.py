"""Models for the four outputs, plus the baselines they must beat.

Four gradient-boosted classifiers (facility, usage day, usage hour, lead days) plus a rule for the
booking clock time (the densest +/-30 min cluster in the resident's own booking times; on validation
it beat boosted regressors, 62% vs 55% within 30 min). The notification ("nudge") is derived:
    nudge day  = (predicted usage day - predicted lead days) mod 7
    nudge time = predicted booking clock time
Hyper-parameters are tuned on the validation split only; the final models are refit on
train + validation and evaluated once on the test split.
"""
import itertools

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import feature_columns

SEED = 42
CLF = {"y_fac": "facility", "y_dow": "usage day", "y_hour": "usage hour", "y_lead": "lead days"}
GRID = [dict(learning_rate=lr, max_leaf_nodes=nl, l2_regularization=l2)
        for lr, nl, l2 in itertools.product([0.05, 0.1], [8, 24], [0.0, 1.0])]


def _make(params):
    return HistGradientBoostingClassifier(max_iter=150, random_state=SEED, early_stopping=False, **params)


def _score(model, X, y):
    return -float(np.mean(model.predict(X) == y))  # lower is better


def clock_fallback(train):
    """Densest +/-30 min booking time across the training rows, used when a resident has no history."""
    ck = train.y_clock.to_numpy(float)
    return float(ck[np.argmax([(np.abs(ck - c) <= 30).sum() for c in ck[::5]]) * 5])


def tune_and_fit(X):
    """Pick params per target on val, refit on train+val. Returns models and the chosen params."""
    cols = feature_columns(X)
    tr, va = X[X.split == "train"], X[X.split == "val"]
    trva = X[X.split.isin(["train", "val"])]
    models, chosen = {}, {}
    for t in CLF:
        scores = [(_score(_make(p).fit(tr[cols], tr[t]), va[cols], va[t]), i) for i, p in enumerate(GRID)]
        best = GRID[min(scores)[1]]
        chosen[t] = {**best, "val_accuracy": -min(scores)[0]}
        models[t] = _make(best).fit(trva[cols], trva[t])
    models["clock_fallback"] = clock_fallback(trva)
    return models, chosen


def predict(models, X):
    cols = feature_columns(X)
    out = pd.DataFrame({"booking_id": X.booking_id.values})
    for t in CLF:
        out[t.replace("y_", "p_")] = models[t].predict(X[cols])
    out["p_clock"] = X.clock_mode.fillna(models["clock_fallback"]).values
    out["p_nudge_dow"] = (out.p_dow - out.p_lead) % 7
    out["p_facility_top3"] = [np.argsort(-r)[:3].tolist() for r in models["y_fac"].predict_proba(X[cols])]
    return out


# ---- baselines (no learning beyond counting) ----------------------------------------------------
def baselines(X, train):
    """Non-learned predictions for the rows of X: global majority, repeat-last-booking, and a
    resident-habit rule (the resident's own recency-weighted mode for each output)."""
    g = {t: train[t].mode()[0] for t in ["y_fac", "y_dow", "y_hour", "y_lead", "y_clock"]}
    g["y_clock"] = float(train.y_clock.median())
    out = {}
    out["majority"] = pd.DataFrame({
        "p_fac": g["y_fac"], "p_dow": g["y_dow"], "p_hour": g["y_hour"], "p_lead": g["y_lead"],
        "p_clock": g["y_clock"]}, index=X.index)
    last = pd.DataFrame({
        "p_fac": X.last_fac.fillna(g["y_fac"]), "p_dow": X.last_dow.fillna(g["y_dow"]),
        "p_hour": X.last_hour.fillna(g["y_hour"]), "p_lead": X.last_lead.fillna(g["y_lead"]),
        "p_clock": X.last_clock.fillna(g["y_clock"])})
    out["repeat_last"] = last
    habit = pd.DataFrame({
        "p_fac": X.top1_fac.fillna(g["y_fac"]), "p_dow": X.top1_fac_dow.fillna(X.dow_mode).fillna(g["y_dow"]),
        "p_hour": X.top1_fac_hour.fillna(X.hour_mode).fillna(g["y_hour"]),
        "p_lead": X.lead_mode.fillna(g["y_lead"]), "p_clock": X.clock_mode.fillna(g["y_clock"])})
    out["resident_habit"] = habit
    for df in out.values():
        df["p_nudge_dow"] = (df.p_dow - df.p_lead) % 7
    return out


def save(models, chosen, path="models/models.joblib"):
    joblib.dump({"models": models, "chosen": chosen}, path)


def load_models(path="models/models.joblib"):
    return joblib.load(path)
