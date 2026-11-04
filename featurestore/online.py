"""Online store: SQLite, one table per view holding the latest row per entity. Filled by materialize()."""
import sqlite3
from pathlib import Path

import duckdb
import pandas as pd

from .offline import TS, check_unique
from .registry import Registry


class OnlineStore:
    def __init__(self, path: str | Path):
        self.con = sqlite3.connect(path)

    def materialize(self, registry: Registry, view_name: str, end_ts: pd.Timestamp) -> int:
        """Upsert the latest feature row per entity with ts <= end_ts. Idempotent; never moves a value back
        in time (an older or repeated run is a no-op). Returns the number of rows read from the offline store."""
        v = registry.get(view_name)
        duck = duckdb.connect()
        check_unique(duck, v)
        rows = duck.execute(
            f"SELECT {v.entity}, {TS}, {', '.join(v.features)} FROM read_parquet(?) WHERE {TS} <= ? "
            f"QUALIFY row_number() OVER (PARTITION BY {v.entity} ORDER BY {TS} DESC) = 1",
            [v.source, end_ts.to_pydatetime()]).fetchall()
        cols = ", ".join(v.features)
        self.con.execute(f"CREATE TABLE IF NOT EXISTS {v.name} ({v.entity} PRIMARY KEY, {TS} TEXT, {cols})")
        updates = ", ".join(f"{c} = excluded.{c}" for c in [TS, *v.features])
        with self.con:  # one transaction: readers never see a half-materialised view
            self.con.executemany(
                f"INSERT INTO {v.name} VALUES ({', '.join('?' * (2 + len(v.features)))}) "
                f"ON CONFLICT({v.entity}) DO UPDATE SET {updates} WHERE excluded.{TS} > {v.name}.{TS}",
                [(r[0], r[1].isoformat(), *r[2:]) for r in rows])  # ISO strings sort chronologically
        return len(rows)

    def get_online_features(self, registry: Registry, refs: list[str], entity_ids: list, now: pd.Timestamp | None = None) -> pd.DataFrame:
        """Feature values for entity_ids at time `now` (default: current UTC time, naive), nulled past the TTL."""
        now = pd.Timestamp.utcnow().tz_localize(None) if now is None else now
        resolved = registry.resolve(refs)
        entity = resolved[0][0].entity
        if any(v.entity != entity for v, _ in resolved):
            raise ValueError("all views in one lookup must share the entity key")
        ids = [i.item() if hasattr(i, "item") else i for i in entity_ids]  # numpy scalars would bind as BLOBs
        out = pd.DataFrame({entity: ids})
        if not ids:
            return out.assign(**{f"{v.name}__{f}": pd.Series(dtype=float) for v, feats in resolved for f in feats})
        for v, feats in resolved:
            marks = ", ".join("?" * len(ids))
            got = pd.read_sql_query(f"SELECT {v.entity}, {TS}, {', '.join(feats)} FROM {v.name} WHERE {v.entity} IN ({marks})",
                                    self.con, params=ids)
            got[TS] = pd.to_datetime(got[TS])
            stale = got[TS] < now - pd.Timedelta(days=v.ttl_days)
            got.loc[stale, feats] = None
            got = got.rename(columns={f: f"{v.name}__{f}" for f in feats}).drop(columns=TS)
            out = out.merge(got, on=entity, how="left")
        return out
