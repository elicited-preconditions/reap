"""Per-predicate-label call/timing/cache-hit bookkeeping.

tampanda's ActionFeasibilityChecker has no instrumentation of its own;
everything here is added in this adapter.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CallStats:
    _calls: dict = field(default_factory=lambda: defaultdict(int))
    _wall_s: dict = field(default_factory=lambda: defaultdict(float))
    _hits: dict = field(default_factory=lambda: defaultdict(int))

    def record_hit(self, label: str) -> None:
        self._hits[label] += 1

    def record_miss(self, label: str, wall_s: float) -> None:
        self._calls[label] += 1
        self._wall_s[label] += wall_s

    def merge(self, other: "CallStats") -> None:
        """Fold another CallStats' counts into this one -- used when a
        second, independent checker instance (e.g. OcclusionChecker,
        which owns its own isolated MuJoCo env) contributed calls that
        belong in the same report."""
        for label in set(other._calls) | set(other._hits):
            self._calls[label] += other._calls[label]
            self._wall_s[label] += other._wall_s[label]
            self._hits[label] += other._hits[label]

    @property
    def total_calls(self) -> int:
        return sum(self._calls.values())

    @property
    def total_wall_s(self) -> float:
        return sum(self._wall_s.values())

    @property
    def total_cache_hits(self) -> int:
        return sum(self._hits.values())

    def by_predicate(self) -> dict:
        labels = set(self._calls) | set(self._hits)
        out = {}
        for label in sorted(labels):
            calls = self._calls[label]
            wall_s = self._wall_s[label]
            out[label] = {
                "calls": calls,
                "wall_s": round(wall_s, 4),
                "avg_ms": round(1000.0 * wall_s / calls, 2) if calls else None,
                "cache_hits": self._hits[label],
            }
        return out

    def summary(self) -> dict:
        return {
            "total_wall_s": round(self.total_wall_s, 4),
            "total_mp_calls": self.total_calls,
            "cache_hits": self.total_cache_hits,
            "by_predicate": self.by_predicate(),
        }
