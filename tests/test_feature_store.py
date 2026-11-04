import numpy as np
import pandas as pd
import pytest

from featurestore.demo_data import build_daily_features, make_transactions
from featurestore.offline import get_historical_features
from featurestore.online import OnlineStore
from featurestore.registry import FeatureView, Registry

T = pd.Timestamp


@pytest.fixture
def reg(tmp_path):
    feats = pd.DataFrame({
        "user_id": [1, 1, 1, 2],
        "event_timestamp": [T("2024-01-01"), T("2024-01-03"), T("2024-01-10"), T("2024-01-02")],
        "score": [10.0, 30.0, 100.0, 7.0],
    })
    feats.to_parquet(tmp_path / "score.parquet")
    prof = pd.DataFrame({"user_id": [1, 2], "event_timestamp": [T("2023-12-01"), T("2024-01-05")], "tier": [1, 2]})
    prof.to_parquet(tmp_path / "profile.parquet")
    r = Registry(tmp_path / "registry.json")
    r.apply(FeatureView("user_score", "user_id", str(tmp_path / "score.parquet"), ["score"], ttl_days=2))
    r.apply(FeatureView("user_profile", "user_id", str(tmp_path / "profile.parquet"), ["tier"], ttl_days=365))
    return r


def test_bad_names_and_tz_aware_sources_rejected(reg, tmp_path):
    with pytest.raises(ValueError):
        FeatureView("bad name", "user_id", "x.parquet", ["f"], ttl_days=1)
    pd.DataFrame({"user_id": [1], "event_timestamp": [T("2024-01-01", tz="UTC")], "x": [1.0]}).to_parquet(tmp_path / "tz.parquet")
    reg.apply(FeatureView("tz", "user_id", str(tmp_path / "tz.parquet"), ["x"], ttl_days=1))
    with pytest.raises(ValueError, match="naive"):
        get_historical_features(reg, pd.DataFrame({"user_id": [1], "event_timestamp": [T("2024-01-02")]}), ["tz:x"])


def test_registry_round_trip(reg, tmp_path):
    again = Registry(tmp_path / "registry.json")
    assert again.get("user_score") == reg.get("user_score")
    with pytest.raises(KeyError):
        again.get("nope")


def test_point_in_time_join_no_leakage(reg):
    labels = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 2, 3],
        "event_timestamp": [T("2024-01-02"), T("2024-01-03"), T("2023-12-31"), T("2024-01-09"), T("2024-01-04"), T("2024-01-04")],
        "label": [0, 1, 0, 1, 1, 0],
    })
    out = get_historical_features(reg, labels, ["user_score:score", "user_profile:tier"])
    assert list(out["label"]) == [0, 1, 0, 1, 1, 0]  # row order kept
    s = out["user_score__score"].tolist()
    assert s[0] == 10.0  # latest before the label, not the 01-03 value (leakage)
    assert s[1] == 30.0  # equal timestamp is visible
    assert pd.isna(s[2])  # label before any feature
    assert pd.isna(s[3])  # latest feature (01-03) is older than ttl=2 days; 01-10 is in the future
    assert s[4] == 7.0
    assert pd.isna(s[5])  # unknown entity
    assert out["user_profile__tier"].tolist()[:4] == [1, 1, 1, 1]  # second view joined independently
    assert pd.isna(out["user_profile__tier"][4])  # user 2 profile appears 01-05, after the 01-04 label


def test_duplicate_feature_rows_rejected(reg, tmp_path):
    dup = pd.DataFrame({"user_id": [1, 1], "event_timestamp": [T("2024-01-01")] * 2, "x": [1.0, 2.0]})
    dup.to_parquet(tmp_path / "dup.parquet")
    reg.apply(FeatureView("dup", "user_id", str(tmp_path / "dup.parquet"), ["x"], ttl_days=1))
    with pytest.raises(ValueError, match="duplicate"):
        get_historical_features(reg, pd.DataFrame({"user_id": [1], "event_timestamp": [T("2024-01-02")]}), ["dup:x"])


def test_materialize_latest_idempotent_and_forward_only(reg, tmp_path):
    store = OnlineStore(tmp_path / "online.db")
    assert store.materialize(reg, "user_score", T("2024-01-05")) == 2
    got = store.get_online_features(reg, ["user_score:score"], [1, 2, 3], now=T("2024-01-05"))
    assert got["user_score__score"][0] == 30.0  # 01-03 row is exactly ttl (2 days) old: still served
    assert pd.isna(got["user_score__score"][1])  # user 2's 01-02 row is 3 days old: expired
    assert pd.isna(got["user_score__score"][2])  # unknown user
    store.materialize(reg, "user_score", T("2024-01-05"))  # idempotent
    store.materialize(reg, "user_score", T("2024-01-02"))  # an older run must not move values back
    got = store.get_online_features(reg, ["user_score:score"], [1], now=T("2024-01-05"))
    assert got["user_score__score"][0] == 30.0
    later = store.get_online_features(reg, ["user_score:score"], [1], now=T("2024-01-09"))
    assert pd.isna(later["user_score__score"][0])  # expired by ttl at read time
    assert store.get_online_features(reg, ["user_score:score"], np.array([1]), now=T("2024-01-05"))["user_score__score"][0] == 30.0
    assert len(store.get_online_features(reg, ["user_score:score"], [], now=T("2024-01-05"))) == 0
    # skew exactly at the TTL boundary: user 1's row is exactly 2 days old at 01-05
    off = get_historical_features(reg, pd.DataFrame({"user_id": [1, 2], "event_timestamp": [T("2024-01-05")] * 2}), ["user_score:score"])
    on = store.get_online_features(reg, ["user_score:score"], [1, 2], now=T("2024-01-05"))
    assert off["user_score__score"].tolist()[0] == on["user_score__score"].tolist()[0] == 30.0
    assert pd.isna(off["user_score__score"][1]) and pd.isna(on["user_score__score"][1])


def test_demo_job_emits_zero_rows_not_gaps():
    txns = pd.DataFrame({"user_id": [7, 8], "ts": [T("2024-01-01 10:00"), T("2024-01-20 10:00")], "amount": [5.0, 1.0]})
    f = build_daily_features(txns)
    u7 = f[f.user_id == 7].set_index("event_timestamp").txn_count_7d
    assert u7[T("2024-01-02")] == 1 and u7[T("2024-01-09")] == 0  # explicit zero once the window is empty
    assert u7.index.max() == T("2024-01-15")  # not scored after 14 days of inactivity


def test_pit_join_matches_brute_force_oracle(tmp_path):
    from featurestore.offline import brute_force_pit

    feats = build_daily_features(make_transactions(n_users=30, days=25, seed=1))
    feats.to_parquet(tmp_path / "f.parquet")
    reg = Registry(tmp_path / "r.json")
    cols = ["txn_count_7d", "amount_sum_7d", "amount_avg_7d"]
    reg.apply(FeatureView("v", "user_id", str(tmp_path / "f.parquet"), cols, ttl_days=2))
    rng = np.random.default_rng(0)  # random label times, including exact feature timestamps (midnights)
    secs = rng.integers(0, 26 * 86400, 300)
    secs[:50] = secs[:50] // 86400 * 86400
    labels = pd.DataFrame({"user_id": rng.integers(0, 33, 300), "event_timestamp": T("2024-01-01") + pd.to_timedelta(secs, unit="s")})
    got = get_historical_features(reg, labels, [f"v:{c}" for c in cols])[[f"v__{c}" for c in cols]].to_numpy(dtype=float)
    want = brute_force_pit(feats, "user_id", labels, cols, ttl_days=2).to_numpy()
    assert np.allclose(got, want, equal_nan=True)


@pytest.mark.parametrize("ts", ["2024-01-08 00:00", "2024-01-15 13:30", "2024-01-29 23:59"])
def test_training_serving_skew_on_demo_data(tmp_path, ts):
    txns = make_transactions(n_users=40, days=30, seed=0)
    build_daily_features(txns).to_parquet(tmp_path / "f.parquet")
    reg = Registry(tmp_path / "r.json")
    feats = ["txn_count_7d", "amount_sum_7d", "amount_avg_7d"]
    reg.apply(FeatureView("user_txn", "user_id", str(tmp_path / "f.parquet"), feats, ttl_days=3))
    refs = [f"user_txn:{f}" for f in feats]
    users = list(range(45))  # includes 5 unknown users
    store = OnlineStore(tmp_path / "o.db")
    store.materialize(reg, "user_txn", T("2024-01-05"))  # an earlier run first: incremental path
    store.materialize(reg, "user_txn", T(ts))
    online = store.get_online_features(reg, refs, users, now=T(ts))
    offline = get_historical_features(reg, pd.DataFrame({"user_id": users, "event_timestamp": [T(ts)] * len(users)}), refs)
    pd.testing.assert_frame_equal(online.reset_index(drop=True), offline[online.columns].reset_index(drop=True),
                                  check_dtype=False)
    assert online.iloc[:, 1:].notna().any().all() and online.iloc[:, 1:].isna().any().all()  # both cases exercised
