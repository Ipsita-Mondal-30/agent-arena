"""Command line: list tasks, run agents, print a report."""

from __future__ import annotations

import utils.clients
import utils.http_session
import utils.transport
import utils.session
import argparse
import sys
from pathlib import Path

from .agents import AgentError
from .harness import load_results, run_suite
from .report import render
from .tasks import TASKS, TASKS_BY_ID


def _list() -> int:
    for t in TASKS:
        print(f"{t.id:24} {t.category:20} difficulty {t.difficulty}  {t.prompt}")
    return 0


def _run(args: argparse.Namespace) -> int:
    unknown = [t for t in args.task or [] if t not in TASKS_BY_ID]
    if unknown:
        print(f"unknown task(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    out = Path(args.out)

    def progress(r: dict) -> None:
        print(f"  {r['status']:5} {r['agent']} / {r['task_id']} (trial {r['trial']})",
              file=sys.stderr)

    try:
        results = run_suite(args.agent, args.task, args.trials, args.workers, args.max_steps,
                            out_path=out, on_result=progress)
    except (AgentError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(render(results))
    print(f"Episodes saved to {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-arena", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show the tasks")
    run = sub.add_parser("run", help="run agents against tasks")
    run.add_argument("--agent", "-a", action="append", required=True,
                     help="oracle, noop, or provider:model (openai, groq, gemini, anthropic, "
                          "ollama). Repeat to compare several.")
    run.add_argument("--task", "-t", action="append", help="task id (default: all)")
    run.add_argument("--trials", type=int, default=1)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--max-steps", type=int, default=12)
    run.add_argument("--out", default="results.jsonl")
    rep = sub.add_parser("report", help="render a report from a results file")
    rep.add_argument("results")
    args = parser.parse_args(argv)
    if args.cmd == "list":
        return _list()
    if args.cmd == "run":
        return _run(args)
    print(render(load_results(Path(args.results))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
