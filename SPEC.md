# mini-feature-store

## Goal
Build "Online/offline feature store with point-in-time joins". Implement a minimal feature store: offline store (Parquet + DuckDB) with point-in-time-correct joins, an online store (SQLite or Redis in docker) with materialisation, a feature-definition registry, and a training-serving skew test proving online and offline values match. Keep it CPU-friendly (the cloud box has no GPU): small models and datasets, a fixed random seed, and a documented runtime.

## Definition of done
- [x] point-in-time join tests pass including leakage cases
- [x] materialisation job works
- [x] skew test passes
- [x] Tests pass with one command; CI workflow runs them
- [x] README with setup, usage, architecture and results
- [x] Self code review done
- [ ] Draft PR open

