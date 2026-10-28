"""Command line entry point:  python -m facility_pred.run {train,evaluate,all}

train     build leakage-safe features, tune on the validation split, refit on train+val, save models
evaluate  score the chronologically later TEST split, write metrics, error analysis and the review workbook
all       data (if missing) -> train -> evaluate
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_gen, evaluate as E, models as M, review
from .features import build, load

DATA, MODELS, RESULTS = Path("data"), Path("models"), Path("results")


def _ensure_data():
    if not (DATA / "bookings.csv").exists():
        df, fac = data_gen.generate()
        DATA.mkdir(exist_ok=True)
        df.to_csv(DATA / "bookings.csv", index=False)
        fac.to_csv(DATA / "facilities.csv", index=False)


def train():
    _ensure_data()
    X = build(load(str(DATA)))
    models, chosen = M.tune_and_fit(X)
    MODELS.mkdir(exist_ok=True)
    M.save(models, chosen, MODELS / "models.joblib")
    (MODELS / "chosen_params.json").write_text(json.dumps(chosen, indent=2))
    n = X.split.value_counts().to_dict()
    print(f"trained on train+val ({n['train']}+{n['val']} rows); params -> models/chosen_params.json")


def _pct(df):
    """Format fractions as percentages; MAE columns (hours / minutes) as plain numbers."""
    out = df.copy().astype(object)
    for i in range(len(df)):
        mae = "mae" in " ".join(map(str, np.atleast_1d(df.index[i]))).lower()
        for j, c in enumerate(df.columns):
            v = df.iloc[i, j]
            if isinstance(v, float):
                out.iloc[i, j] = f"{v:.2f}" if mae or "mae" in str(c) else f"{v:.1%}"
    return out


def evaluate():
    bookings = load(str(DATA))
    X = build(bookings)
    fac = pd.read_csv(DATA / "facilities.csv").set_index("facility_id").facility_name.to_dict()
    saved = M.load_models(MODELS / "models.joblib")
    test, trainval = X[X.split == "test"], X[X.split != "test"]
    P = M.predict(saved["models"], test).set_index(test.index)
    h = E.hits(test, P)
    top3 = P.p_facility_top3.tolist()
    model_m = E.metrics(test, P, top3)
    base_m = {n: E.metrics(test, b) for n, b in M.baselines(test, trainval).items()}

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps({"model": model_m, "baselines": base_m, "n_test": len(test)}, indent=2))

    rows = {"model (gradient boosting)": model_m, **{f"baseline: {k}": v for k, v in base_m.items()}}
    comp = pd.DataFrame({n: {"Facility acc": m["facility"]["accuracy"], "Day acc": m["day"]["accuracy"],
                             "Hour acc": m["hour"]["accuracy"], "Nudge acc": m["nudge"]["accuracy"],
                             "Mean k of 4": m["overall"]["mean_k_of_4"], "All 4 match": m["overall"]["all_4_match"]}
                         for n, m in rows.items()}).T
    per_out = pd.DataFrame({o: m for o, m in model_m.items() if o != "overall"}).T.dropna(how="all", axis=1)
    by_hist, by_fac = E.segment_table(test, h)
    by_fac.index = by_fac.index.map(lambda i: fac[f"F{int(i) + 1}"])
    conf = pd.crosstab(test.y_fac.map(lambda i: fac[f"F{int(i) + 1}"]), P.p_fac.map(lambda i: fac[f"F{int(i) + 1}"]),
                       rownames=["actual"], colnames=["predicted"])
    overall = pd.DataFrame({"Value": model_m["overall"]}).rename(
        index={"mean_k_of_4": "Mean k of 4", "all_4_match": "All 4 match"})
    per_out = per_out.stack().rename("Value").to_frame()
    per_out.index.names = ["Output", "Metric"]
    comp.index.name = "Method"
    by_hist = by_hist.rename(columns=str.title).rename(columns={"N": "Bookings", "Mean_K_Of_4": "Mean k of 4"})
    by_hist.index.name = "Prior bookings"
    by_fac = by_fac.rename(columns=str.title).rename(columns={"N": "Bookings"})
    by_fac.index.name = "Actual facility"
    blocks = [(f"Overall (test split, {len(test)} bookings)", overall),
              ("Per-output metrics", per_out),
              ("Model vs baselines", comp),
              ("By amount of resident history (hit rates)", by_hist), ("By actual facility (hit rates)", by_fac),
              ("Facility confusion (counts)", conf)]
    table = review.review_table(test, P, h, bookings, fac)
    Path("outputs").mkdir(exist_ok=True)
    review.write_workbook("outputs/prediction_review.xlsx", table, blocks)
    table.to_csv("outputs/prediction_review.csv", index=False)

    md = ["# Test-split results\n", f"{len(test)} bookings from {test.booking_ts.min():%Y-%m-%d} onward.\n"]
    md += [f"## {t}\n\n```\n{_pct(d).to_string()}\n```\n" for t, d in blocks]
    (RESULTS / "results.md").write_text("\n".join(md))
    print(comp.map(lambda v: f"{v:.1%}").to_string())
    print("\nwrote results/metrics.json, results/results.md, outputs/prediction_review.xlsx")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["train", "evaluate", "all"])
    s = ap.parse_args().step
    if s in ("train", "all"):
        train()
    if s in ("evaluate", "all"):
        evaluate()


if __name__ == "__main__":
    main()
