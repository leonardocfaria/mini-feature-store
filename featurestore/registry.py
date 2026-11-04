"""Feature definitions, persisted to one JSON file: the single source of truth for both stores."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeatureView:
    name: str
    entity: str  # join key column, e.g. "user_id"
    source: str  # Parquet with columns: entity, event_timestamp, *features
    features: list[str]
    ttl_days: float  # a feature row older than this at lookup time is treated as missing

    def __post_init__(self):  # runs on apply() and on load: names become SQL identifiers
        if not all(n.isidentifier() for n in [self.name, self.entity, *self.features]):
            raise ValueError("view, entity and feature names must be identifiers")


class Registry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.views = {}
        if self.path.exists():
            self.views = {v["name"]: FeatureView(**v) for v in json.loads(self.path.read_text())}

    def apply(self, view: FeatureView) -> None:
        self.views[view.name] = view
        self.path.write_text(json.dumps([asdict(v) for v in self.views.values()], indent=2))

    def get(self, name: str) -> FeatureView:
        if name not in self.views:
            raise KeyError(f"unknown feature view {name!r}")
        return self.views[name]

    def resolve(self, refs: list[str]) -> list[tuple[FeatureView, list[str]]]:
        """['view:feature', ...] -> [(view, [features])] in first-seen order."""
        out: dict[str, list[str]] = {}
        for ref in refs:
            name, feat = ref.split(":")
            if feat not in self.get(name).features:
                raise KeyError(f"{name} has no feature {feat!r}")
            out.setdefault(name, []).append(feat)
        return [(self.get(n), feats) for n, feats in out.items()]
