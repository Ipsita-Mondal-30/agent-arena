"""Turn episode results into a markdown report."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

Result = dict[str, Any]


def _pct(passed: int, total: int) -> str:
    return f"{100 * passed / total:.0f}%" if total else "n/a"


def summarize(results: list[Result]) -> dict[str, dict[str, Any]]:
    """Per-agent numbers. Infrastructure errors are counted but left out of the pass rate."""
    by_agent: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_agent[r["agent"]].append(r)
    summary = {}
    for agent, rows in by_agent.items():
        graded = [r for r in rows if r["status"] != "error"]
        passed = sum(r["status"] == "pass" for r in graded)
        per_task: dict[str, list[bool]] = defaultdict(list)
        for r in graded:
            per_task[r["task_id"]].append(r["status"] == "pass")
        summary[agent] = {
            "episodes": len(rows), "graded": len(graded), "passed": passed,
            "errors": len(rows) - len(graded),
            "pass_rate": passed / len(graded) if graded else 0.0,
            # share of tasks solved in every trial: the reliability number
            "all_trials": (sum(all(v) for v in per_task.values()) / len(per_task)
                           if per_task else 0.0),
            "avg_steps": sum(r["steps"] for r in graded) / len(graded) if graded else 0.0,
            "tool_error_rate": (sum(r["tool_errors"] for r in graded)
                                / max(1, sum(r["steps"] for r in graded))),
        }
    return summary


def render(results: list[Result], max_failures: int = 5) -> str:
    if not results:
        return "No results.\n"
    agents = sorted({r["agent"] for r in results})
    summary = summarize(results)
    out = ["# Results", "", "## Overall", "",
           "| Agent | Pass rate | Solved in every trial | Avg steps | Tool-error rate "
           "| Infra errors |",
           "|---|---|---|---|---|---|"]
    for a in agents:
        s = summary[a]
        out.append(f"| {a} | {_pct(s['passed'], s['graded'])} ({s['passed']}/{s['graded']}) "
                   f"| {100 * s['all_trials']:.0f}% | {s['avg_steps']:.1f} "
                   f"| {100 * s['tool_error_rate']:.0f}% | {s['errors']} |")

    out += ["", "## By category", "", "| Category | " + " | ".join(agents) + " |",
            "|---|" + "---|" * len(agents)]
    for cat in sorted({r["category"] for r in results}):
        cells = []
        for a in agents:
            rows = [r for r in results if r["agent"] == a and r["category"] == cat
                    and r["status"] != "error"]
            cells.append(_pct(sum(r["status"] == "pass" for r in rows), len(rows)))
        out.append(f"| {cat} | " + " | ".join(cells) + " |")

    out += ["", "## By task", "", "| Task | Difficulty | " + " | ".join(agents) + " |",
            "|---|---|" + "---|" * len(agents)]
    tasks = sorted({(r["difficulty"], r["task_id"]) for r in results})
    for difficulty, task_id in tasks:
        cells = []
        for a in agents:
            rows = [r for r in results if r["agent"] == a and r["task_id"] == task_id
                    and r["status"] != "error"]
            ok = sum(r["status"] == "pass" for r in rows)
            cells.append(f"{ok}/{len(rows)}" if rows else "-")
        out.append(f"| {task_id} | {difficulty} | " + " | ".join(cells) + " |")

    failures = [r for r in results if r["status"] == "fail"]
    if failures:
        out += ["", "## Failures", ""]
        for a in agents:
            mine = [r for r in failures if r["agent"] == a][:max_failures]
            if not mine:
                continue
            out.append(f"### {a}")
            for r in mine:
                out.append(f"- **{r['task_id']}** (trial {r['trial']}): "
                           + "; ".join(r["problems"]))
            out.append("")
    errors = [r for r in results if r["status"] == "error"]
    if errors:
        out += ["## Infrastructure errors", ""]
        out += [f"- {r['agent']} / {r['task_id']}: {r['error']}" for r in errors[:max_failures]]
    return "\n".join(out) + "\n"
