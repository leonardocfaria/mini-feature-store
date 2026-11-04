"""End-to-end demo: raw txns -> daily features (offline) -> PIT training set -> materialise -> online lookup -> skew check.

python run_demo.py [--out demo_out]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from featurestore.demo_data import START, build_daily_features, make_transactions
from featurestore.offline import brute_force_pit, get_historical_features
from featurestore.online import OnlineStore
from featurestore.registry import FeatureView, Registry

FEATURES = ["txn_count_7d", "amount_sum_7d", "amount_avg_7d"]


def main(out: str = "demo_out", seed: int = 0) -> dict:
    out = Path(out)
    out.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    txns = make_transactions(seed=seed)
    feats = build_daily_features(txns)
    feats.to_parquet(out / "user_txn.parquet")
    reg = Registry(out / "registry.json")
    reg.apply(FeatureView("user_txn", "user_id", str(out / "user_txn.parquet"), FEATURES, ttl_days=3))
    refs = [f"user_txn:{f}" for f in FEATURES]

    # Training set: label = "user transacts in the next 24h", observed at random times. PIT join features.
    rng = np.random.default_rng(seed)
    labels = pd.DataFrame({"user_id": rng.integers(0, 200, 2000),
                           "event_timestamp": START + pd.to_timedelta(rng.integers(7 * 86400, 59 * 86400, 2000), unit="s")})
    nxt = txns.groupby("user_id").ts.apply(np.array)
    labels["label"] = [int(any((t > ts) & (t <= ts + pd.Timedelta(days=1)) for t in nxt.get(u, [])))
                       for u, ts in zip(labels.user_id, labels.event_timestamp)]
    t_join = time.perf_counter()
    train = get_historical_features(reg, labels, refs)
    t_train = time.perf_counter()

    # Leakage audit: compare with a brute-force point-in-time lookup (feature_ts <= label_ts, within TTL).
    oracle = brute_force_pit(feats, "user_id", labels, FEATURES, ttl_days=3)
    got = train[[f"user_txn__{f}" for f in FEATURES]].to_numpy(dtype=float)
    pit_mismatches = int((~np.isclose(got, oracle.to_numpy(), equal_nan=True)).sum())

    # Serving: materialise "now" and compare online lookups with an offline query at the same instant.
    now = pd.Timestamp("2024-02-20 12:00")
    store = OnlineStore(out / "online.db")
    n_mat = store.materialize(reg, "user_txn", now)
    users = list(range(200))
    t1 = time.perf_counter()
    online = store.get_online_features(reg, refs, users, now=now)
    lookup_ms = (time.perf_counter() - t1) * 1000
    offline = get_historical_features(reg, pd.DataFrame({"user_id": users, "event_timestamp": [now] * len(users)}), refs)
    skew = int((~((online[offline.columns[2:]] == offline[offline.columns[2:]]) |
                  (online[offline.columns[2:]].isna() & offline[offline.columns[2:]].isna()))).to_numpy().sum())
    summary = {
        "transactions": len(txns), "feature_rows": len(feats), "training_rows": len(train),
        "training_rows_with_features": int(train[f"user_txn__{FEATURES[0]}"].notna().sum()),
        "label_rate": round(float(labels.label.mean()), 3), "pit_mismatches_vs_brute_force": pit_mismatches,
        "pit_join_seconds_2000_rows": round(t_train - t_join, 3), "materialized_entities": n_mat,
        "online_lookup_ms_200_entities": round(lookup_ms, 2),
        "online_entities_fresh": int(online[f"user_txn__{FEATURES[0]}"].notna().sum()),
        "skew_mismatches": skew,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="demo_out")
    main(p.parse_args().out)
