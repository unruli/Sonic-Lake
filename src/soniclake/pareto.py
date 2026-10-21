from __future__ import annotations

from typing import Any, Iterable


def pareto_frontier(
    rows: Iterable[dict[str, Any]],
    minimize: tuple[str, ...],
    maximize: tuple[str, ...],
) -> list[dict[str, Any]]:
    values = list(rows)
    frontier: list[dict[str, Any]] = []
    for candidate in values:
        dominated = False
        for challenger in values:
            if challenger is candidate:
                continue
            no_worse = all(challenger[key] <= candidate[key] for key in minimize) and all(
                challenger[key] >= candidate[key] for key in maximize
            )
            strictly_better = any(challenger[key] < candidate[key] for key in minimize) or any(
                challenger[key] > candidate[key] for key in maximize
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier

