"""Sanity report for the synthetic dataset: checks it has the properties the brief asks for."""
import argparse

import pandas as pd


def report(data_dir="data"):
    b = pd.read_csv(f"{data_dir}/bookings.csv", parse_dates=["booking_ts", "usage_ts"])
    fac = pd.read_csv(f"{data_dir}/facilities.csv").set_index("facility_id").facility_name
    b["lead_h"] = (b.usage_ts - b.booking_ts).dt.total_seconds() / 3600
    per_res = b.groupby("resident_id").size()

    print(f"bookings={len(b)}  residents={b.resident_id.nunique()}  "
          f"span={b.booking_ts.min().date()}..{b.usage_ts.max().date()}")
    print("\nFacility share (imbalance):")
    print((b.facility_id.value_counts(normalize=True).rename(index=fac) * 100).round(1).to_string())
    print("\nSparsity: bookings per resident")
    print(per_res.describe().round(1).to_string())
    print(f"residents with <10 bookings: {(per_res < 10).mean():.0%}")
    print("\nUsage hour (top 8):", b.usage_ts.dt.hour.value_counts().head(8).to_dict())
    print("Usage weekday share:", (b.usage_ts.dt.dayofweek.value_counts(normalize=True)
                                   .sort_index().round(2)).to_dict())
    print("\nLead time (hours):")
    print(b.lead_h.describe().round(1).to_string())

    top = b.groupby("resident_id").facility_id.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    print(f"\nMean share of a resident's favourite facility: {top[per_res >= 10].mean():.0%} (preferences)")

    b["half"] = (b.booking_ts >= b.booking_ts.min() + (b.booking_ts.max() - b.booking_ts.min()) / 2)
    mix = b.groupby(["half", "facility_id"]).size().unstack(0, fill_value=0)
    print("\nMonthly volume (seasonality/growth):", b.groupby(b.usage_ts.dt.month).size().to_dict())

    pool = b[b.facility_id == "F2"].groupby(b.usage_ts.dt.month).size()
    print("Pool bookings by month:", pool.to_dict())

    assert (b.booking_ts < b.usage_ts).all(), "booking must precede usage"
    print("\nOK: all booking timestamps precede usage timestamps")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    report(ap.parse_args().data)
