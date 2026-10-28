from facility_pred.data_gen import generate


def test_deterministic_and_valid():
    a, _ = generate(seed=1, n_residents=20)
    b, _ = generate(seed=1, n_residents=20)
    assert a.equals(b)
    assert (a.booking_ts < a.usage_ts).all()
    assert a.booking_id.is_unique
