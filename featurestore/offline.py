"""Offline store: Parquet + DuckDB ASOF joins for point-in-time-correct training data."""
import duckdb
import pandas as pd

from .registry import FeatureView, Registry

TS = "event_timestamp"


def check_unique(con, view: FeatureView) -> None:
    """One row per (entity, timestamp), otherwise 'latest value' is ambiguous."""
    dup = con.execute(f"SELECT count(*) FROM (SELECT 1 FROM read_parquet(?) GROUP BY {view.entity}, {TS} HAVING count(*) > 1)",
                      [view.source]).fetchone()[0]
    if dup:
        raise ValueError(f"{view.name}: {dup} duplicate ({view.entity}, {TS}) rows in {view.source}")


def brute_force_pit(features: pd.DataFrame, entity: str, entity_df: pd.DataFrame, cols: list[str], ttl_days: float) -> pd.DataFrame:
    """Slow, obviously-correct reference for one view: used to test get_historical_features."""
    rows = []
    for key, ts in zip(entity_df[entity], entity_df[TS]):
        f = features[(features[entity] == key) & (features[TS] <= ts) & (features[TS] >= ts - pd.Timedelta(days=ttl_days))]
        rows.append(f.loc[f[TS].idxmax(), cols].tolist() if len(f) else [None] * len(cols))
    return pd.DataFrame(rows, columns=cols, dtype=float)


def get_historical_features(registry: Registry, entity_df: pd.DataFrame, refs: list[str]) -> pd.DataFrame:
    """For each row (entity, event_timestamp) take, per view, the latest feature row with
    feature_ts <= event_timestamp (no future data), nulled if older than the view's TTL.
    Returns entity_df's columns plus '<view>__<feature>' columns, in entity_df's row order."""
    con = duckdb.connect()
    con.register("entity_df", entity_df.assign(__row=range(len(entity_df))))
    select, joins = ["e.*"], []
    for i, (view, feats) in enumerate(registry.resolve(refs)):
        check_unique(con, view)
        f = f"f{i}"
        fresh = f"{f}.{TS} >= e.{TS} - to_microseconds(CAST({view.ttl_days} * 86400e6 AS BIGINT))"
        select += [f"CASE WHEN {fresh} THEN {f}.{c} END AS {view.name}__{c}" for c in feats]
        src = view.source.replace("'", "''")
        joins.append(f"ASOF LEFT JOIN read_parquet('{src}') {f} ON e.{view.entity} = {f}.{view.entity} AND e.{TS} >= {f}.{TS}")
    sql = f"SELECT {', '.join(select)} FROM entity_df e {' '.join(joins)} ORDER BY e.__row"
    return con.execute(sql).df().drop(columns="__row")
