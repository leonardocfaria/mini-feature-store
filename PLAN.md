# mini-feature-store Implementation Plan

## Brainstorm (answered from SPEC + repo evidence)
- **What is it?** A minimal Feast-like feature store that gets the hard parts right: point-in-time-correct training data (no label leakage), an online store filled by materialisation, and proof that online and offline agree (no training/serving skew).
- **Data:** synthetic, seeded. Users make transactions, and a daily batch job computes per-user features (`txn_count_7d`, `amount_sum_7d`, `amount_avg_7d`). They are stored as Parquet with an `event_timestamp`.
- **Offline store:** Parquet + DuckDB. A historical retrieval is one `ASOF LEFT JOIN` per feature view: for each (entity, label time) take the latest feature row with `feature_ts <= label_ts`, then null it if older than the view's TTL. "Equal timestamp is visible" is a documented choice: a feature computed *at* t is known at t.
- **Online store:** SQLite (stdlib, no Redis container needed). One table per feature view: `entity_id PRIMARY KEY, event_timestamp, <features>`.
- **Materialisation:** `materialize(view, end_ts)` takes the latest row per entity with `ts <= end_ts` and upserts it only if newer than what is stored (idempotent and incremental). Rows older than the TTL at `end_ts` are not served.
- **Registry:** `FeatureView` dataclasses persisted to a JSON file (name, entity, source path, features, ttl). The registry is the single source of truth used by both stores, which is what prevents skew.
- **Skew test:** materialise at T, then compare `get_online_features(ids)` with `get_historical_features(ids at T)` for every entity and feature. They must be equal, including nulls for expired or unknown entities.
- **Non-goals:** streaming ingestion, on-demand transforms, created-timestamp tie-breaks (one row per entity per timestamp is assumed and validated), a server API.

## Files
- `featurestore/registry.py`: `FeatureView`, `Registry` (apply/get/list, JSON-persisted).
- `featurestore/offline.py`: `get_historical_features(registry, entity_df, refs)`.
- `featurestore/online.py`: `OnlineStore(path)`, `materialize(registry, view, end_ts)`, `get_online_features(...)`.
- `featurestore/demo_data.py`: `make_transactions(seed)`, `build_daily_features(txns)`.
- `run_demo.py`: registry → offline training set → materialise → online lookup → skew check.

## Tasks (TDD)
- [x] 1. registry round-trip, duplicate-key validation.
- [x] 2. PIT join: feature after the label is never used (leakage); equal timestamp used; before first feature → null; TTL expiry → null; two views joined; label rows kept in order.
- [x] 3. materialise: latest ≤ end; idempotent; incremental only moves forward; expired rows are not served.
- [x] 4. skew test on demo data at several T; demo script; README; CI.
