"""The suite checks itself: every reference solution passes and every trap fails."""

import pytest

from agent_arena.agents import NoopAgent, ScriptedAgent
from agent_arena.harness import run_episode
from agent_arena.tasks import TASKS, has_amount, has_any

TRAPS = [(t.id, name) for t in TASKS for name in t.traps]


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_reference_solution_passes(task):
    result = run_episode(ScriptedAgent("oracle"), task.id)
    assert result["status"] == "pass", result["problems"]


@pytest.mark.parametrize(("task_id", "trap"), TRAPS)
def test_trap_is_rejected(task_id, trap):
    result = run_episode(ScriptedAgent(trap), task_id)
    assert result["status"] == "fail", f"grader accepted trap {trap}"
    assert result["problems"]


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_doing_nothing_never_passes(task):
    assert run_episode(NoopAgent(), task.id)["status"] == "fail"


def test_every_task_has_at_least_one_trap():
    assert all(t.traps for t in TASKS)


def test_suite_covers_varied_categories():
    assert {"state-change", "read-only", "safety", "impossible-request", "ambiguity"} <= {
        t.category for t in TASKS}


def test_has_amount_formats():
    assert has_amount("total is 19000 cents", 19000)
    assert has_amount("that is $190.00", 19000)
    assert has_amount("1,900.00 dollars", 190000)
    assert not has_amount("total is 9000", 19000)
    assert not has_amount("order 119000", 19000)
    assert not has_amount("190.005", 19000)


def test_has_any_is_case_insensitive():
    assert has_any("YES it can", r"\byes\b")
    assert not has_any("eyes", r"\byes\b")
