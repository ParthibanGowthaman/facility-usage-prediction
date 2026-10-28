import numpy as np
import pandas as pd

from facility_pred.data_gen import generate
from facility_pred.features import build, feature_columns, split_of, TRAIN_END, VAL_END


def _data():
    df, _ = generate(seed=3, n_residents=25)
    df["fac"] = df.facility_id.str[1:].astype(int) - 1
    return df.sort_values(["booking_ts", "booking_id"]).reset_index(drop=True)


def _same(a, b):
    return a.reset_index(drop=True).equals(b.reset_index(drop=True)) or np.allclose(
        a.to_numpy(float), b.to_numpy(float), equal_nan=True)


def test_features_ignore_the_row_itself_and_the_future():
    b = _data()
    X = build(b)
    cols = feature_columns(X)
    k = len(b) // 2
    # Scramble the facility and usage time of every booking from row k onward.
    b2 = b.copy()
    b2.loc[k:, "fac"] = (b2.loc[k:, "fac"] + 3) % 6
    b2.loc[k:, "usage_ts"] = b2.loc[k:, "usage_ts"] + pd.Timedelta(days=2, hours=5)
    X2 = build(b2)
    # Rows before k must be identical, and row k's own features must not see row k's data.
    assert _same(X.loc[:k, cols], X2.loc[:k, cols])


def test_first_booking_of_each_resident_has_no_history():
    X = build(_data())
    first = X.groupby("resident_id").head(1)
    assert (first.n_prev == 0).all()
    assert first[[c for c in feature_columns(X) if c != "n_prev"]].isna().all().all()


def test_split_is_chronological_and_disjoint():
    X = build(_data())
    tr, va, te = (X[X.split == s] for s in ("train", "val", "test"))
    assert tr.booking_ts.max() < TRAIN_END <= va.booking_ts.min()
    assert va.booking_ts.max() < VAL_END <= te.booking_ts.min()
    assert len(tr) + len(va) + len(te) == len(X)
    assert set(split_of(pd.Series([TRAIN_END, VAL_END]))) == {"val", "test"}
