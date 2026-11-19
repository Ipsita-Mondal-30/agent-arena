"""State diffing and the strict "exactly these changes, nothing else" check used by graders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .db import Snapshot

Row = dict[str, Any]


@dataclass
class Diff:
    updated: dict[tuple[str, Any], tuple[Row, Row]] = field(default_factory=dict)
    inserted: dict[str, list[Row]] = field(default_factory=dict)
    deleted: list[tuple[str, Any]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.updated or self.deleted or any(self.inserted.values()))


def compute(before: Snapshot, after: Snapshot) -> Diff:
    diff = Diff()
    for table, old_rows in before.items():
        new_rows = after[table]
        for pk, old in old_rows.items():
            if pk not in new_rows:
                diff.deleted.append((table, pk))
            elif new_rows[pk] != old:
                diff.updated[(table, pk)] = (old, new_rows[pk])
        added = [row for pk, row in new_rows.items() if pk not in old_rows]
        if added:
            diff.inserted[table] = added
    return diff


def check(
    diff: Diff,
    updates: dict[tuple[str, Any], Row] | None = None,
    inserts: dict[str, list[Callable[[Row], bool]]] | None = None,
) -> list[str]:
    """Return the problems with `diff`; an empty list means it is exactly what was expected.

    `updates` maps (table, pk) to the columns that must change and their new values. Every other
    column of that row must stay as it was. `inserts` maps a table to one predicate per expected
    new row. Anything not listed (an extra update, extra row, or any delete) is a problem, so a
    correct run cannot hide collateral damage.
    """
    updates = updates or {}
    inserts = inserts or {}
    problems: list[str] = []

    for key, wanted in updates.items():
        if key not in diff.updated:
            problems.append(f"expected {key[0]} {key[1]} to change, but it did not")
            continue
        old, new = diff.updated[key]
        for column, value in wanted.items():
            if new[column] != value:
                problems.append(
                    f"{key[0]} {key[1]}.{column} is {new[column]!r}, expected {value!r}")
        for column in old:
            if column not in wanted and old[column] != new[column]:
                problems.append(f"{key[0]} {key[1]}.{column} changed unexpectedly")
    for key in diff.updated:
        if key not in updates:
            problems.append(f"unexpected change to {key[0]} {key[1]}")

    for table, predicates in inserts.items():
        remaining = list(diff.inserted.get(table, []))
        for index, predicate in enumerate(predicates):
            match = next((row for row in remaining if predicate(row)), None)
            if match is None:
                problems.append(f"no new {table} row matched expectation #{index + 1}")
            else:
                remaining.remove(match)
    for table, rows in diff.inserted.items():
        expected = len(inserts.get(table, []))
        if len(rows) > expected:
            problems.append(f"{len(rows) - expected} unexpected new row(s) in {table}")

    for table, pk in diff.deleted:
        problems.append(f"{table} {pk} was deleted")
    return problems
