"""Runs agents against tasks, in parallel, and records every episode."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .agents import Agent, AgentError, Transcript, make_agent
from .env import SupportDeskEnv
from .tasks import TASKS, TASKS_BY_ID

Result = dict[str, Any]


def run_episode(agent: Agent, task_id: str, trial: int = 0, max_steps: int = 12) -> Result:
    """Play one task. status is 'pass', 'fail', or 'error' (infrastructure, not the agent)."""
    task = TASKS_BY_ID[task_id]
    env = SupportDeskEnv(max_steps=max_steps)
    obs, _ = env.reset(task_id)
    agent.begin(task_id)
    transcript: Transcript = [{"role": "user", "content": obs["prompt"]}]
    result: Result = {
        "agent": agent.name, "task_id": task_id, "category": task.category,
        "difficulty": task.difficulty, "trial": trial, "status": "error", "problems": [],
        "steps": 0, "tool_errors": 0, "answer": "", "error": None, "seconds": 0.0,
    }
    started = time.monotonic()
    try:
        while True:
            action = agent.act(obs["system"], transcript, obs["tools"])
            if action.get("type") != "final":
                call_id = f"call_{len(transcript)}"
                name, args = str(action.get("name") or "invalid"), action.get("arguments", {})
                transcript.append({"role": "assistant", "type": "tool", "id": call_id,
                                   "name": name, "arguments": args})
            else:
                transcript.append({"role": "assistant", "type": "final",
                                   "content": str(action.get("answer", ""))})
            obs_next, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                result.update(status="pass" if reward == 1.0 else "fail",
                              problems=info["problems"], answer=info["answer"])
                break
            transcript.append({"role": "tool", "id": call_id, "name": name,
                               "content": json.dumps(obs_next)})
    except AgentError as exc:
        result.update(status="error", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - a crashing agent must not stop the whole run
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    result.update(steps=env.steps, tool_errors=env.tool_errors,
                  seconds=round(time.monotonic() - started, 2), transcript=transcript)
    return result


def run_suite(
    specs: Iterable[str],
    task_ids: Iterable[str] | None = None,
    trials: int = 1,
    workers: int = 4,
    max_steps: int = 12,
    factory: Callable[[str], Agent] = make_agent,
    out_path: Path | None = None,
    on_result: Callable[[Result], None] | None = None,
) -> list[Result]:
    specs = list(specs)
    ids = list(task_ids) if task_ids else [t.id for t in TASKS]
    for spec in specs:  # fail fast on missing keys or unknown providers
        factory(spec)
    jobs = [(spec, tid, trial) for spec in specs for tid in ids for trial in range(trials)]
    results: list[Result] = []
    out = out_path.open("w") if out_path else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(lambda s=s, t=t, k=k: run_episode(factory(s), t, k, max_steps))
                       for s, t, k in jobs]
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                if out:
                    out.write(json.dumps(res) + "\n")
                    out.flush()
                if on_result:
                    on_result(res)
    finally:
        if out:
            out.close()
    results.sort(key=lambda r: (r["agent"], r["task_id"], r["trial"]))
    return results


def load_results(path: Path) -> list[Result]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
