"""Seeded synthetic transactions and the daily batch job that turns them into feature rows."""
import numpy as np
import pandas as pd

START = pd.Timestamp("2024-01-01")


def make_transactions(n_users: int = 200, days: int = 60, seed: int = 0) -> pd.DataFrame:
    """Each user is active for a random stretch of days (so some go quiet: TTL matters), ~Poisson txns/day."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        first = rng.integers(0, days // 2)
        last = rng.integers(first + 1, days + 1)
        rate = rng.uniform(0.3, 3)
        for d in range(first, last):
            for _ in range(rng.poisson(rate)):
                ts = START + pd.Timedelta(days=int(d), seconds=int(rng.integers(0, 86400)))
                rows.append((u, ts, round(float(rng.lognormal(3, 1)), 2)))
    return pd.DataFrame(rows, columns=["user_id", "ts", "amount"]).sort_values(["ts", "user_id"], ignore_index=True)


def build_daily_features(txns: pd.DataFrame) -> pd.DataFrame:
    """At each midnight D, for users with a txn in [D-7d, D): count, sum, mean of amounts. Timestamp = D,
    i.e. the time the batch job could have produced the value (it only uses data strictly before D)."""
    days = pd.date_range(txns.ts.min().normalize() + pd.Timedelta(days=1), txns.ts.max().normalize() + pd.Timedelta(days=1))
    out = []
    for d in days:
        w = txns[(txns.ts >= d - pd.Timedelta(days=7)) & (txns.ts < d)]
        g = w.groupby("user_id").amount.agg(["count", "sum", "mean"]).reset_index()
        g.columns = ["user_id", "txn_count_7d", "amount_sum_7d", "amount_avg_7d"]
        out.append(g.assign(event_timestamp=d))
    return pd.concat(out, ignore_index=True)[["user_id", "event_timestamp", "txn_count_7d", "amount_sum_7d", "amount_avg_7d"]]
