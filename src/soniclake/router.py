from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RoutedQuery:
    family: str
    representation: str
    sample_id: str
    measured_execution_s: float
    router_ns: int


class CatalogRouter:
    """Deterministic router over a selected physical-design catalog.

    The router executes the control path and joins each request to a measured
    record. It intentionally does not rerun expensive frozen models; callers
    should describe aggregate results as measured trace replay.
    """

    def __init__(
        self,
        assignment: Mapping[str, str],
        measurements: Mapping[tuple[str, str], Sequence[Mapping]],
        lexical_lookup_s: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        if not assignment:
            raise ValueError("catalog assignment cannot be empty")
        self._assignment = dict(assignment)
        self._measurements = measurements
        self._lexical_lookup_s = lexical_lookup_s or {}
        self._positions: dict[tuple[str, str], int] = {}
        for family, representation in self._assignment.items():
            if (representation, family) not in measurements:
                raise ValueError(f"missing measurements for {representation}/{family}")

    def route(self, family: str) -> RoutedQuery:
        start = perf_counter_ns()
        representation = self._assignment[family]
        key = (representation, family)
        rows = self._measurements[key]
        position = self._positions.get(key, 0)
        row = rows[position % len(rows)]
        self._positions[key] = position + 1
        if family == "lexical" and representation in self._lexical_lookup_s:
            lookups = self._lexical_lookup_s[representation]
            execution_s = float(row["decode_s"]) + float(lookups[position % len(lookups)])
        else:
            execution_s = float(row["decode_s"]) + float(row["query_s"])
        elapsed = perf_counter_ns() - start
        return RoutedQuery(
            family=family,
            representation=representation,
            sample_id=str(row["sample_id"]),
            measured_execution_s=execution_s,
            router_ns=elapsed,
        )
