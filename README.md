# mini-feature-store

A minimal feature store that gets the hard parts right:
- **point-in-time-correct** training data, so no label leakage;
- an **online store** filled by an idempotent **materialisation** job;
- a **feature registry** shared by both stores;
- a **training/serving skew test** proving online and offline values match.

## Setup
```bash
cd ml/mini-feature-store
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python -m pytest -q          # 8 tests, ~1.5 s
python run_demo.py           # end-to-end demo, ~1 s, writes demo_out/ (parquet, registry.json, online.db, summary.json)
```
```python
from featurestore.registry import FeatureView, Registry
from featurestore.offline import get_historical_features
from featurestore.online import OnlineStore

reg = Registry("registry.json")
reg.apply(FeatureView("user_txn", "user_id", "user_txn.parquet",
                      ["txn_count_7d", "amount_sum_7d"], ttl_days=3))
train = get_historical_features(reg, labels_df, ["user_txn:txn_count_7d"])   # labels_df: user_id, event_timestamp, ...
store = OnlineStore("online.db")
store.materialize(reg, "user_txn", end_ts=pd.Timestamp("2024-02-20"))
store.get_online_features(reg, ["user_txn:txn_count_7d"], [1, 2, 3])
```

## Architecture
```
raw txns ─► daily batch job ─► user_txn.parquet (user_id, event_timestamp, features)   ◄── registry.json
                                    │                                                     (views, entity,
        offline: DuckDB ASOF LEFT JOIN per view                                            source, ttl)
          latest row with feature_ts <= label_ts, nulled if older than ttl  ─► training set
                                    │
        materialize(end_ts): latest row per entity with ts <= end_ts (QUALIFY row_number)
          ─► SQLite upsert, only if newer (idempotent, never moves back) ─► online.db
                                    │
        get_online_features(ids, now): primary-key lookup, nulled if older than ttl at `now`
```
Semantics (the same in both stores, which is why there is no skew):
- A feature row is visible at time t if `feature_ts <= t`. A value computed at t is known at t, so equal timestamps are visible. Anything later is the future and is never joined.
- A feature row is fresh if `feature_ts >= t - ttl`. Otherwise the value is null rather than stale.
- `(entity, event_timestamp)` must be unique per view. Both stores check this, and duplicates raise an error instead of returning an arbitrary value.
- Timestamps are naive UTC.

Tests:
- **Leakage cases:**
  - a feature row after the label is not used;
  - an equal timestamp is used;
  - a label before any feature gets null;
  - a feature older than the TTL gets null;
  - an unknown entity gets null;
  - a second view is joined independently;
  - row order is kept.
- **Oracle:** 300 random label times (50 of them exactly on feature timestamps) are joined and compared with a brute-force pandas implementation.
- **Materialisation:** it picks the latest row; re-running is a no-op; an older run never moves values back; the TTL applies at read time.
- **Skew:** synthetic demo data at 3 serving times (incl. exactly midnight, the TTL boundary). The test materialises incrementally (an earlier run first), then requires online == offline for 45 entities × 3 features, covering both fresh and null cases.

Choices: SQLite instead of Redis keeps the online store in the standard library, with no container to run. The registry is a JSON file. View and feature names must be identifiers, because they become SQL identifiers.

## Results (demo, seed 0)
| metric | value |
|---|---|
| synthetic transactions → daily feature rows | 7495 → 5455 (200 users, 60 days) |
| PIT training set | 2000 labelled rows, 1094 with fresh features, 0.044 s DuckDB join |
| mismatches vs brute-force PIT oracle | **0** |
| materialised entities | 198 (76 still fresh at serving time, ttl 3 days) |
| online lookup, 200 entities | ~11 ms (SQLite, one IN query) |
| online vs offline skew mismatches | **0** |

**Debugging note.** The first leakage audit in `run_demo.py` reported 213 leaking rows. It found each joined row's source by matching on feature values, but 874 feature rows share all their values with another day of the same user, because the 7-day window can be unchanged from one day to the next. So the audit itself was ambiguous. A brute-force oracle confirmed 0 mismatches, and it now replaces that audit in the demo and in a unit test.

## Not done (time box)
Streaming/push ingestion, on-demand transforms, created-timestamp tie-breaks, a Redis backend, an HTTP serving API, schema/dtype validation of sources.
